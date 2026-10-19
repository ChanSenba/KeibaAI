"""KeibaAI の操作画面(Streamlit)。`keibaai-gui` で起動する。"""
from __future__ import annotations

import contextlib
import datetime as dt
import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from keibaai import betting, db, factor_view, gui, past_races, progress
from keibaai import gui_help
from keibaai.gui_help import TIPS
from keibaai.cli import reparse_cache
from keibaai.fetch import Fetcher

st.set_page_config(page_title="KeibaAI", page_icon=":material/insights:", layout="wide")
st.logo(str(Path(__file__).with_name("logo.svg")), size="large")

# テーマで足りない部分のスタイル(画面幅への対応もここ)
st.html(Path(__file__).with_name("style.css"))
# light-dark() の色を、OS ではなく Streamlit で選ばれているテーマに合わせる
if st.context.theme.type in ("light", "dark"):
    st.html(f"<style>:root {{ color-scheme: {st.context.theme.type}; }}</style>")

if "db_path" not in st.session_state:
    st.session_state["db_path"] = "data/keiba.db"
    st.session_state["cache_dir"] = "data/cache"


def header(eyebrow: str, title: str, lead: str) -> None:
    st.html(f'<div class="kb-eyebrow">{eyebrow}</div>')
    st.title(title, anchor=False)
    st.html(f'<div class="kb-lead">{lead}</div>')


def connection():
    return db.connect(st.session_state["db_path"])


class LogBox:
    """print の出力を画面のログ欄に流す。"""

    def __init__(self, max_lines: int = 200):
        self.box = st.empty()
        self.lines: list[str] = []
        self.max_lines = max_lines

    def write(self, s: str) -> int:
        for line in s.splitlines():
            if line.strip():
                self.lines.append(line)
        if self.lines:
            self.box.code("\n".join(self.lines[-self.max_lines:]), language=None, height=220)
        return len(s)

    def flush(self) -> None:
        pass


def bar_chart(df, x: str, y: str, horizontal: bool = False):
    enc = dict(
        x=alt.X(f"{y}:Q", title=None) if horizontal else alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=0, labelOverlap="greedy", labelSeparation=12)),
        y=alt.Y(f"{x}:N", title=None, sort="-x") if horizontal else alt.Y(f"{y}:Q", title=None),
        tooltip=[alt.Tooltip(f"{x}:N", title=""), alt.Tooltip(f"{y}:Q", title="レース数")],
    )
    mark = alt.Chart(df).mark_bar(cornerRadiusEnd=4) if horizontal else alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    return mark.encode(**enc).properties(height=260, padding={"left": 8, "top": 6, "right": 6, "bottom": 0})


# ---------------------------------------------------------------- データ確認
def page_overview() -> None:
    header("Overview", "取得済みデータ", "DB に入っているレース結果の量と期間です。")
    conn = connection()
    s = gui.summary(conn)
    if s["races"] == 0:
        with st.container(border=True):
            st.markdown("#### まだデータがありません")
            st.write("「取得」で期間を選んで取得すると、ここに件数とグラフが出ます。")
            st.page_link(PAGES["collect"], label="取得へ", icon=":material/arrow_forward:")
        return

    monthly = gui.races_by_month(conn)
    cols = st.container(key="kpis").columns(4)
    cols[0].metric("レース", f"{s['races']:,}", border=True, height="stretch", help=TIPS["kpi_races"])
    cols[1].metric("出走", f"{s['results']:,}", border=True, height="stretch", help=TIPS["kpi_results"])
    cols[2].metric("馬", f"{s['horses']:,}", border=True, height="stretch", help=TIPS["kpi_horses"])
    first, last = dt.date.fromisoformat(s["first"]), dt.date.fromisoformat(s["last"])
    cols[3].metric("期間", f"{(last - first).days // 30 + 1} か月", border=True,
                   delta=f"{first:%Y/%m} – {last:%Y/%m}", delta_color="off", delta_arrow="off",
                   height="stretch", help=TIPS["kpi_span"])

    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        st.markdown("##### 月別レース数", help=TIPS["monthly"])
        st.altair_chart(bar_chart(monthly, "month", "races"), width="stretch")
    with right.container(border=True):
        st.markdown("##### 競馬場別", help=TIPS["venues"])
        st.altair_chart(bar_chart(gui.races_by_venue(conn), "venue", "races", horizontal=True), width="stretch")

    with st.container(border=True):
        st.markdown("##### 最近のレース", help=TIPS["recent"])
        h = gui_help.RECENT_COLUMNS
        st.dataframe(gui.recent_races(conn), hide_index=True, width="stretch", height=380, column_config={
            "date": st.column_config.DateColumn("日付", format="YYYY/MM/DD", help=h["date"]),
            "venue": st.column_config.TextColumn("競馬場", help=h["venue"]),
            "round": st.column_config.NumberColumn("R", format="%dR", help=h["round"]),
            "name": st.column_config.TextColumn("レース名", help=h["name"]),
            "course_type": st.column_config.TextColumn("芝/ダ", help=h["course_type"]),
            "distance": st.column_config.NumberColumn("距離", format="%dm", help=h["distance"]),
            "ground": st.column_config.TextColumn("馬場", help=h["ground"]),
            "runners": st.column_config.NumberColumn("頭数", help=h["runners"])})


# ---------------------------------------------------------------- データ取得
def _apply_preset() -> None:
    name = st.session_state.get("preset")
    if name:
        st.session_state["start"], st.session_state["end"] = gui.preset_range(name, dt.date.today())


def require_jv() -> bool:
    """成績・出馬表は JRA-VAN(JV-Link)から取る。JV-Link を呼べない PC では、その旨を出して False を返す。"""
    from keibaai import jvlink

    if jvlink.available():
        return True
    st.error("この PC では JV-Link を呼べないため、取得できません(32bit Python と JV-Link のセットアップが要ります)。")
    return False


def page_collect() -> None:
    header("Collect", "データ取得",
           "選んだ期間の JRA レース結果を JRA-VAN(JV-Link)から取得します。"
           "取得済みのレースは飛ばすので、同じ期間を何度実行しても大丈夫です。")
    today = dt.date.today()
    st.session_state.setdefault("start", gui.preset_range("先週末", today)[0])
    st.session_state.setdefault("end", gui.preset_range("先週末", today)[1])

    with st.container(border=True):
        ok = require_jv()
        st.segmented_control("よく使う期間", gui.PRESETS, key="preset", on_change=_apply_preset)
        c1, c2 = st.columns(2)
        start = c1.date_input("開始日", key="start", format="YYYY/MM/DD", max_value=today)
        end = c2.date_input("終了日", key="end", format="YYYY/MM/DD", max_value=today)
        days = gui.race_days(start, end) if start <= end else []
        if start > end:
            st.error("終了日が開始日より前になっています。")
        else:
            st.caption(f":material/event: 対象 {len(days)} 日 ・ :material/schedule: 1〜数分(JV-Link からまとめて取ります)")
        run = st.button("取得を開始", type="primary", icon=":material/download:", disabled=not days or not ok,
                        width="stretch")

    st.caption(":material/info: 取得中に別の画面へ移ったりボタンを押したりすると止まります。"
               "それまでに取得した分は保存されています。")

    if run:
        conn = connection()
        with st.status("取得しています…", expanded=True) as status:
            bar = st.progress(0.0)
            log = LogBox()

            def on_day(i, n, d):
                bar.progress(i / n if n else 1.0, f"{d:%Y/%m/%d} ({i + 1}/{n} 日目)" if d else "完了")

            try:
                with contextlib.redirect_stdout(log):
                    from keibaai import jvlink

                    bar.progress(0.1, "JV-Link から書き出しています")
                    total, _ = jvlink.fetch_results(conn, start, end)
                    bar.progress(1.0, "完了")
            except Exception as e:  # 連続失敗(アクセス制限の可能性)や通信エラー。途中までの分は保存済み
                status.update(label="取得が止まりました", state="error")
                st.error(str(e))
            else:
                status.update(label=f"完了しました。新しく {total} レースを保存しました", state="complete")


# ---------------------------------------------------------------- 取得の進み具合
@st.cache_data(ttl=3600, show_spinner=False)
def _progress_total(db_path: str, kind: str) -> int | None:
    return progress.read_total(db_path, progress.KINDS[kind][2])


@st.cache_data(ttl=3600, show_spinner=False)
def _data_status(db_path: str):
    return progress.read_data_status(db_path)


def _status_badge(p) -> str:
    """取得の状態の短い表示。"""
    if p.alive:
        return ":green[:material/play_circle: 動いています]"
    if p.last_saved is None:
        return ":gray[:material/hourglass_empty: まだ始まっていません]"
    if p.total and p.done >= p.total:
        return ":green[:material/check_circle: 揃っています]"
    return ":orange[:material/pause_circle: 止まっています]"


def _show_data_grid() -> None:
    """データの揃い具合を1つの枠にまとめる(取得元が JRA-VAN でも netkeiba でも同じ並び)。"""
    s = _data_status(st.session_state["db_path"])
    tr = progress.progress_of("training", st.session_state["cache_dir"],
                              _progress_total(st.session_state["db_path"], "training"))
    with st.container(border=True):
        st.markdown("#### データの揃い具合", help=TIPS["data_grid"])
        if s is None:
            st.caption("DB を読めませんでした(取り込み中の可能性があります)。少しあとで開き直してください。")
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("成績", f"{s.races:,} レース", help=f"最新 {s.last_race_date or '-'}(JRA-VAN)")
        frac = s.pedigree_fraction
        c2.metric("血統", f"{frac:.0%}" if frac is not None else "-",
                  help=f"成績に出てくる {s.horses:,} 頭のうち {s.horses_with_pedigree:,} 頭(JRA-VAN)")
        c3.metric("調教タイム", f"約{s.training_rows / 10000:,.0f}万本", help=f"最新 {s.last_training_date or '-'}(JRA-VAN)")
        c4.metric("調教の評価", f"{tr.fraction:.0%}" if tr.fraction is not None else f"{tr.done:,} 件",
                  help=f"{tr.done:,} / {tr.total or '-'} レース(短評と A〜D。netkeiba)")
        c4.markdown(_status_badge(tr))
        if tr.alive and tr.fraction is not None:
            st.progress(tr.fraction, f"調教の評価 {tr.done:,} / {tr.total:,}")
        bits = [f"成績の最新 {s.last_race_date or '-'}", f"調教タイムの最新 {s.last_training_date or '-'}"]
        if tr.last_saved:
            bits.append(f"調教の評価の最後の保存 {tr.last_saved:%m/%d %H:%M}")
        if tr.per_min:
            bits.append(f"速さ 約{tr.per_min:.1f}件/分")
        if tr.eta_hours is not None and tr.alive:
            bits.append(f"あと約{tr.eta_hours:.0f}時間")
        st.caption(" ・ ".join(bits))


def page_progress() -> None:
    header("Progress", "取得の進み具合",
           "成績・血統・調教がどこまで揃っているかと、バックグラウンドの取得の進み具合です。")

    @st.fragment(run_every=30)
    def body() -> None:
        _show_data_grid()
        logs = sorted(Path("data").glob("out-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:1]
        if logs:
            with st.expander(f"ログの最後 ({logs[0].name})"):
                st.code(progress.tail(logs[0]) or "(空)", language=None)
        st.caption(f":material/autorenew: 30秒ごとに更新 ・ {dt.datetime.now():%H:%M:%S}")

    body()


# ---------------------------------------------------------------- 再解析
def page_reparse() -> None:
    header("Reparse", "再解析",
           "保存済みのページを解析し直して DB を更新します。通信はしません。プログラムを更新したあとに実行してください。")
    n_cached = len(list((Path(st.session_state["cache_dir"]) / "race").glob("*.html")))
    with st.container(border=True):
        st.metric("保存済みページ", f"{n_cached:,}")
        run = st.button("再解析を実行", type="primary", icon=":material/refresh:", disabled=n_cached == 0,
                        width="stretch")
    if run:
        with st.status("解析しています…", expanded=False) as status:
            log = LogBox()
            with contextlib.redirect_stdout(log):
                ok, bad = reparse_cache(connection(), st.session_state["cache_dir"])
            status.update(label=f"完了しました。{ok:,} レースを更新、{bad:,} 件は飛ばしました",
                          state="complete", expanded=bad > 0)


# ---------------------------------------------------------------- 学習・評価
MODEL_NOTES = {
    "基準": "オッズだけで予測。的中・回収率は毎回1番人気を単勝で買った場合",
    "オッズなし": "馬・騎手・調教師の過去成績とレース条件だけで予測",
    "オッズあり": "上の情報に単勝オッズ・人気を加えて予測",
}


def _note(name: str) -> str:
    return next((v for k, v in MODEL_NOTES.items() if name.startswith(k)), "")


def page_train() -> None:
    header("Model", "学習・評価", "評価開始日より前のレースで学習し、それ以後のレースで成績を確かめます。")
    conn = connection()
    first, last = gui.date_bounds(conn)
    if first is None:
        with st.container(border=True):
            st.markdown("#### まだデータがありません")
            st.page_link(PAGES["collect"], label="取得へ", icon=":material/arrow_forward:")
        return

    with st.container(border=True):
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        earliest, default = gui.eval_start_bounds(first, last)
        test_from = c1.date_input("評価開始日", default, min_value=earliest, max_value=last, format="YYYY/MM/DD",
                                  help=f"取得済み期間は {first:%Y/%m/%d} – {last:%Y/%m/%d} です。"
                                       f"学習用に、評価開始日より前のレースが要るので、選べるのは {earliest:%Y/%m/%d} 以降です")
        run = c2.button("学習して評価", type="primary", icon=":material/model_training:", width="stretch")
    if run:
        try:
            with st.spinner("学習しています(データ量によって数分かかります)"):
                st.session_state["report"] = (test_from, gui.train_reports(conn, test_from.isoformat()))
        except ValueError:
            st.error(f"{test_from:%Y/%m/%d} では、学習用か評価用のレースが足りません。"
                     f"{earliest:%Y/%m/%d} から {last:%Y/%m/%d} の間の日付を選んでください"
                     f"(取得済みの期間は {first:%Y/%m/%d} から {last:%Y/%m/%d})。")

    if "report" not in st.session_state:
        return
    when, report = st.session_state["report"]
    st.subheader("評価結果", anchor=False, help=TIPS["train_result"])
    st.caption(f"{when:%Y/%m/%d} 以降の {int(report['レース数'].iloc[0]):,} レースで評価")
    best_roi = report["単勝回収率(%)"].max()
    for col, (_, r) in zip(st.container(key="models").columns(len(report), gap="small"), report.iterrows()):
        with col.container(border=True, height="stretch"):
            badge = " :green-badge[回収率 最高]" if r["単勝回収率(%)"] == best_roi else ""
            st.markdown(f"**{r['モデル']}**{badge}")
            st.caption(_note(r["モデル"]))
            auc = "—" if r["勝AUC"] != r["勝AUC"] else f"{r['勝AUC']:.3f}"
            stats = [("1番手的中", f"{r['1番手的中']:.1%}"), ("単勝回収率", f"{r['単勝回収率(%)']:.1f}%"), ("勝AUC", auc)]
            st.html('<div class="kb-stats">' + "".join(
                f'<div><div class="kb-stat-label">{k}</div><div class="kb-stat-value">{v}</div></div>'
                for k, v in stats) + "</div>")

    with st.expander("すべての指標", icon=":material/table:"):
        h = gui_help.TRAIN_COLUMNS
        st.dataframe(report, hide_index=True, width="stretch", column_config={
            "モデル": st.column_config.TextColumn(help=h["モデル"]),
            "レース数": st.column_config.NumberColumn(help=h["レース数"]),
            "勝LogLoss": st.column_config.NumberColumn(format="%.4f", help=h["勝LogLoss"]),
            "勝AUC": st.column_config.NumberColumn(format="%.3f", help=h["勝AUC"]),
            "複AUC": st.column_config.NumberColumn(format="%.3f", help=h["複AUC"]),
            "1番手的中": st.column_config.NumberColumn(format="percent", help=h["1番手的中"]),
            "単勝回収率(%)": st.column_config.NumberColumn(format="%.1f", help=h["単勝回収率(%)"]),
        })


# ---------------------------------------------------------------- 予想
PRED_COLUMNS = {
    "mark": st.column_config.TextColumn(
        "印", width="small",
        help="◎○▲△は勝率の高い順(オッズがあればオッズ込み、なければ能力のみ)。☆は印のない馬のうち、期待値(オッズ込み。なければ能力のみ)がいちばん高く1を超える穴馬"),
    "star": st.column_config.TextColumn(
        "★", width="small",
        help="オッズ込みのモデルが、オッズより高く評価している馬(その期待値1.1以上・勝率3%以上・単勝30倍以内)。"
             "過去8年の検証では、これらの馬を単勝で買った回収率は90〜97%で、100%には届かず、買えば得になる印ではありません。"
             "表示される期待値は、実際より1〜3割高く出る傾向があります"),
    "number": st.column_config.NumberColumn("馬番", format="%d", help="馬番(ゼッケンの番号)"),
    "horse_name": st.column_config.TextColumn("馬名", help="馬の名前"),
    "jockey": st.column_config.TextColumn("騎手", help="今回乗る騎手"),
    "popularity": st.column_config.NumberColumn("人気", format="%d", help="単勝オッズの低い順の人気"),
    "win_odds": st.column_config.NumberColumn("単勝", format="%.1f", help="単勝オッズ(倍)。100円で当たると、この倍率が払い戻される"),
    "p_win_ability": st.column_config.ProgressColumn(
        "勝率(能力)", min_value=0, max_value=1, format="percent",
        help="オッズを見ずに、過去の成績などの能力だけで出した勝率。オッズが無くても出る"),
    "p_win_market": st.column_config.ProgressColumn(
        "勝率(オッズ込)", min_value=0, max_value=1, format="percent",
        help="能力の勝率と、オッズから読み取れる市場の確率を、過去の結果で決めた割合で混ぜた勝率"),
    "p_win_quality": st.column_config.ProgressColumn("馬の質", min_value=0, max_value=1, format="percent",
                                                     help="過去の成績・上がり・距離適性・クラスだけで見た勝率"),
    "p_win_fit": st.column_config.ProgressColumn("適性", min_value=0, max_value=1, format="percent",
                                                 help="距離・芝ダート・競馬場・馬場での過去の成績だけで見た勝率"),
    "p_win_rotation": st.column_config.ProgressColumn("ローテ", min_value=0, max_value=1, format="percent",
                                                      help="レース間隔・休み明け何戦目・連闘だけで見た勝率"),
    "p_win_human": st.column_config.ProgressColumn("騎手・厩舎", min_value=0, max_value=1, format="percent",
                                                   help="騎手と調教師の成績・乗り替わりだけで見た勝率"),
    "p_win_track": st.column_config.ProgressColumn("展開・馬場", min_value=0, max_value=1, format="percent",
                                                   help="枠順・脚質・直近の馬場の偏り(先行/内有利)だけで見た勝率"),
    "p_win_pedigree": st.column_config.ProgressColumn("血統", min_value=0, max_value=1, format="percent",
                                                      help="父・母の父の産駒の、同じ条件での過去の成績だけで見た勝率(血統を取得した馬のみ)"),
    "p_win_training": st.column_config.ProgressColumn("調教", min_value=0, max_value=1, format="percent",
                                                      help="最終追い切りの評価(短評とA〜D)だけで見た勝率(調教を取得したレースのみ)"),
    "ev_ability": st.column_config.NumberColumn(
        "期待値(能力)", format="%.2f",
        help="勝率(能力) × 単勝オッズ。1を超えると、勝率が正しければ長い目で得になる目安。大穴ほど大きく出やすい"),
    "ev_market": st.column_config.NumberColumn(
        "期待値(オッズ込)", format="%.2f",
        help="勝率(オッズ込) × 単勝オッズ。1を超えると、勝率が正しければ長い目で得になる目安"),
}
ENTRY_COLUMNS = {
    "bracket": st.column_config.NumberColumn("枠", format="%d", help="枠番(帽子の色)"),
    "number": PRED_COLUMNS["number"], "horse_name": PRED_COLUMNS["horse_name"],
    "sex_age": st.column_config.TextColumn("性齢", help="性別と年齢(牡=牡馬、牝=牝馬、セ=せん馬)"),
    "weight_carried": st.column_config.NumberColumn("斤量", format="%.1f", help="騎手が背負う重さ(kg)"),
    "jockey": PRED_COLUMNS["jockey"],
    "win_odds": PRED_COLUMNS["win_odds"], "popularity": PRED_COLUMNS["popularity"],
}


SCOPES = ["その日の全レース", "1レースだけ"]


def page_predict() -> None:
    header("Predict", "予想",
           "これから走るレースの出馬表とオッズを取得し、各馬の勝率・期待値と予想印を表示します。")
    today = dt.date.today()
    conn = connection()
    with st.container(border=True):
        date = st.date_input("対象日", gui.next_race_day(today), format="YYYY/MM/DD", key="pred_date")
        info = gui.entries_summary(conn, date.isoformat())
        if info["races"]:
            st.caption(f":material/check_circle: 取得済み {info['races']} レース・{info['horses']} 頭"
                       f"(オッズあり {info['with_odds']} 頭) ・ 取得 {info['fetched_at']}")
        else:
            st.caption(":material/info: この日の出馬表はまだ取得していません")
        ok = require_jv()
        single = st.segmented_control(
            "範囲", SCOPES, default=SCOPES[0], key="pred_scope", width="stretch") == SCOPES[1]
        st.caption(":material/info: 出馬表は JRA-VAN から、その日の全レースをまとめて取ります(1〜3分)。"
                   "1レースだけのときは、オッズをそのレースだけ取ります。馬番は金曜の枠順の発表後、オッズは発売開始後に入ります。")
        if not single:
            c1, c2 = st.columns(2)
            fetch = c1.button("全レースの出馬表とオッズを取得", icon=":material/download:", width="stretch", disabled=not ok)
            predict = c2.button("予想する", type="primary", icon=":material/insights:", width="stretch",
                                disabled=info["races"] == 0)

    picked = None
    if single:
        picked, fetch, predict = single_race_controls(conn, date)
        if picked is None:
            return
    st.caption(":material/info: オッズは発売開始後に出ます。オッズが更新されたら、もう一度取得してください。")

    if fetch:
        fetch_entries(conn, date, None if picked is None else [picked["race_id"]])

    if predict:
        try:
            with st.spinner("学習して予想しています(数分かかります)"):
                st.session_state["prediction"] = (date, gui.add_marks(gui.predict_table(conn, date)))
            with st.spinner("予想の見解に使う、各要素の値を計算しています(初めてのときは1〜2分かかります)"):
                _race_factor_values(conn, date, None)
        except ValueError as e:
            st.error(str(e))

    table = None
    if "prediction" in st.session_state:
        when, table = st.session_state["prediction"]
        if when != date:
            st.info(f"{when:%Y/%m/%d} の予想です。{date:%Y/%m/%d} を見るには「予想する」を押してください。")
            table = None
    if table is not None and picked is not None and picked["race_id"] not in set(table["race_id"]):
        table = None  # 予想のあとで取得したレースは、もう一度「予想する」まで予想がない

    if picked is not None:
        if picked["fetched"]:
            st.subheader("出馬表", anchor=False)
            with st.container(border=True):
                race_detail(conn, gui.day_races(conn, date.isoformat()).set_index("race_id").loc[picked["race_id"]],
                            picked["race_id"], table, date)
        return

    if table is not None:
        st.subheader("期待値の高い馬", anchor=False, help=TIPS["top_ev"])
        key = "ev_market" if table["ev_market"].notna().any() else "ev_ability"
        top = table.dropna(subset=[key]).nlargest(8, key)
        courses = gui.day_courses(conn, date.isoformat())  # 予想の表にはコースの列がないので出馬表から引く
        with st.container(border=True):
            st.dataframe(top.assign(race=top["venue"] + top["round"].astype(int).astype(str) + "R",
                       course=top["race_id"].map(courses).fillna(""))[
                ["race", "course", "mark", "star", "number", "horse_name", "win_odds", "p_win_ability", "p_win_market",
                 "ev_ability", "ev_market"]].style.map(mark_style, subset=["mark"]),
                hide_index=True, width="stretch", column_config={
                    **PRED_COLUMNS, "race": st.column_config.TextColumn("レース", help="競馬場とレース番号"),
                    "course": st.column_config.TextColumn("コース", help="芝・ダート、回り、距離")})

    races = gui.day_races(conn, date.isoformat())
    if not races.empty:
        st.subheader("出馬表" if table is None else "レース別", anchor=False)
        race, box = race_picker(races, key=f"board_{date}")
        with box:
            race_detail(conn, race, race["race_id"], table, date)


def single_race_controls(conn, date: dt.date):
    """1レースだけのとき: レースを選ぶ欄と、そのレースの取得・予想ボタン。選んだレースと押されたボタンを返す。

    レース一覧は JRA-VAN の出馬表(entry_races)から作る。まだなければ、出馬表だけを先に取る(オッズは取らない)。
    """
    from keibaai import jvlink

    choices = gui.race_choices(conn, date.isoformat())
    if choices.empty:
        with st.container(border=True):
            st.caption(":material/info: この日のレース一覧をまだ読み込んでいません。")
            if st.button("レース一覧を読み込む", icon=":material/list:", type="primary", disabled=not jvlink.available()):
                with st.spinner("JRA-VAN から出馬表を読み込んでいます(1〜3分)"):
                    try:
                        n = jvlink.fetch_entries(conn, date.isoformat(), odds=False)
                    except Exception as e:
                        st.error(str(e))
                        return None, False, False
                if not n:
                    st.warning("この日の出馬表はまだありません(馬番は金曜の枠順の発表後に入ります)。")
                    return None, False, False
                st.rerun()
        return None, False, False
    picked, box = race_picker(choices, key=f"pick_{date}")
    with box:
        if picked["fetched"]:
            st.caption(f":material/check_circle: {picked['venue']}{picked['round']}R の出馬表は取得済みです")
        else:
            st.caption(f":material/info: {picked['venue']}{picked['round']}R の出馬表はまだ取得していません")
        c1, c2 = st.columns(2)
        fetch = c1.button("このレースの出馬表とオッズを取得", icon=":material/download:", width="stretch")
        predict = c2.button("このレースを予想", type="primary", icon=":material/insights:", width="stretch",
                            disabled=not picked["fetched"])
    return picked, fetch, predict


def fetch_entries(conn, date: dt.date, race_ids: list[str] | None) -> None:
    """JRA-VAN から出馬表とオッズを取る。race_ids を渡すと、オッズはそのレースだけ。"""
    with st.status("取得しています…", expanded=True) as status:
        bar = st.progress(0.0)
        log = LogBox()
        try:
            with contextlib.redirect_stdout(log):
                from keibaai import jvlink

                bar.progress(0.1, "JV-Link から書き出しています")
                n = jvlink.fetch_entries(conn, date.isoformat(), race_ids=race_ids)
                if not n:
                    print("馬番がまだ確定していないか、出馬表がまだ出ていません(馬番は枠順の発表後に入ります)")
        except Exception as e:
            status.update(label="取得が止まりました", state="error")
            st.error(str(e))
        else:
            bar.progress(1.0, "完了")
            status.update(label=f"完了しました。{n} レースの出馬表を保存しました", state="complete")
            st.rerun() if n else None


# 枠番の色(JRA の帽子の色)
BRACKET_COLORS = {1: ("#ffffff", "#222222"), 2: ("#222222", "#ffffff"), 3: ("#e8383d", "#ffffff"),
                  4: ("#2f6fdb", "#ffffff"), 5: ("#ffd400", "#222222"), 6: ("#2e9e5b", "#ffffff"),
                  7: ("#f39800", "#222222"), 8: ("#f19ec2", "#222222")}


def bracket_color(b) -> str:
    bg, fg = BRACKET_COLORS.get(b, (None, None))
    return f"background-color: {bg}; color: {fg}; text-align: center" if bg else ""


def mark_style(m) -> str:
    return "font-weight: 700; text-align: center" if m else ""


def race_picker(races, key: str):
    """競馬場のタブと 1R〜12R のボタンでレースを選ぶ(netkeiba のレース一覧に近い形)。

    選んだレースの行と、ボタンの下に続きを書ける枠を返す。
    """
    venues = list(dict.fromkeys(races["venue"]))
    with st.container(key="venue_tabs"):
        venue = st.segmented_control("競馬場", venues, default=venues[0], key=f"venue_{key}",
                                     label_visibility="collapsed", width="stretch") or venues[0]
    here = races[races["venue"] == venue]
    rounds = here["round"].astype(int).tolist()
    box = st.container(border=True, key="race_board")
    with box, st.container(key="round_grid"):
        rnd = st.segmented_control("レース", rounds, default=rounds[0], format_func=lambda r: f"{r}R",
                                   key=f"round_{key}_{venue}", label_visibility="collapsed",
                                   width="stretch") or rounds[0]
    return here[here["round"] == rnd].iloc[0], box


def race_head(race) -> None:
    """レース番号・名前・条件の見出し。"""
    st.html(gui.race_head_html(race))


def _show_gate_bias(conn, race, date: dt.date) -> None:
    """このコース・この A〜E コースでの、内・中・外の枠の過去の 3 着内率(参考)。"""
    table = factor_view.gate_bias(conn, race, date.isoformat())
    if table.empty:
        return
    with st.expander("枠の傾向(参考)", icon=":material/view_column:"):
        st.caption("内・中・外の枠の、過去の 3 着内率", help=TIPS["gate_bias"])
        h = gui_help.GATE_COLUMNS
        pct = lambda c: st.column_config.NumberColumn(format="percent", help=h[c])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "レース数": st.column_config.NumberColumn(format="%d", help=h["レース数"]),
            "内枠": pct("内枠"), "中枠": pct("中枠"), "外枠": pct("外枠"),
            "内−外": st.column_config.NumberColumn("内−外(pt)", format="%+.1f", help="内枠 − 外枠(ポイント)。プラスなら内の枠のほうが 3 着以内に来ている")})


def _show_same_day(conn, race, date: dt.date) -> None:
    """同じ日の前のレース(同じ競馬場・芝ダ)の展開。芝の状態や内外の有利不利は、後のレースほど変わる。"""
    today = factor_view.same_day_history(conn, race, date.isoformat())
    if today.empty:
        return
    st.markdown("**同じ日の前のレース**")
    st.markdown(factor_view.same_day_text(race.get("venue"), race.get("course_type"), today))
    shown = today.drop(columns=["race_id", "日付"])
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "R": st.column_config.NumberColumn("R", format="%d"),
        "前半3F": st.column_config.NumberColumn(format="%.1f"), "後半3F": st.column_config.NumberColumn(format="%.1f"),
        "ペース差": st.column_config.NumberColumn(format="%+.1f", help="前半3F − 後半3F。プラスは後傾(スロー)、マイナスはハイペース"),
        "先行有利度": st.column_config.NumberColumn(format="percent", help="前に付けた馬の3着内率 − ほかの馬の率"),
        "内有利度": st.column_config.NumberColumn(format="percent", help="内の馬(馬番が前の3分の1)の3着内率 − 外の馬(後ろの3分の1)の率"),
        "勝ち馬": st.column_config.NumberColumn("勝ち馬(馬番)", format="%d"),
        "勝ち馬の位置": st.column_config.NumberColumn(format="%.2f", help="勝ち馬の最初のコーナーの位置 ÷ 頭数。0に近いほど前")})


def _show_race_flow(conn, race, one, date: dt.date) -> None:
    """このレースの展開の見立て(メンバーの脚質構成)と、同じコースの直近の展開。予想の展開適性の元になる。"""
    history = factor_view.course_history(conn, race, date.isoformat())
    flow = factor_view.race_flow(one, history)
    with st.expander("レース展開(脚質構成と同コースの直近)", icon=":material/timeline:", expanded=True):
        st.markdown(flow["text"], help=TIPS["race_flow"])
        names = {k: " ".join(str(n) for n in v) for k, v in flow["lineup"].items() if v}
        st.html('<div class="kb-marks">' + "".join(
            f'<span class="kb-mark"><b>{html.escape(k)}</b>{html.escape(v)}</span>' for k, v in names.items()) + "</div>")
        if not history.empty:
            shown = history.drop(columns=["race_id", "内有利度", "勝ち馬"]).copy()
            shown["日付"] = pd.to_datetime(shown["日付"]).dt.strftime("%m/%d")
            h = gui_help.FLOW_COLUMNS
            st.dataframe(shown, hide_index=True, width="stretch", column_config={
                "R": st.column_config.NumberColumn("R", format="%d", help=h["R"]),
                "前半3F": st.column_config.NumberColumn(format="%.1f", help=h["前半3F"]),
                "後半3F": st.column_config.NumberColumn(format="%.1f", help=h["後半3F"]),
                "ペース差": st.column_config.NumberColumn(format="%+.1f", help="前半3F − 後半3F。プラスは後傾(スロー)、マイナスはハイペース"),
                "先行有利度": st.column_config.NumberColumn(format="percent", help="前に付けた馬の3着内率 − ほかの馬の率"),
                "勝ち馬の位置": st.column_config.NumberColumn(format="%.2f", help="勝ち馬の最初のコーナーの位置 ÷ 頭数。0に近いほど前")})
        _show_same_day(conn, race, date)


def race_detail(conn, race, race_id: str, prediction, date: dt.date | None = None) -> None:
    """レース名・条件と出馬表。予想があれば印・勝率・期待値も出す。"""
    race_head(race)
    one = gui.race_entries(conn, race_id, prediction)
    if "mark" in one:
        marked = one[one["mark"] != ""].sort_values("mark", key=lambda m: m.map(MARK_ORDER.get))
        st.html('<div class="kb-marks">' + "".join(
            f'<span class="kb-mark"><b>{m}</b>{n} {html.escape(str(h))}</span>'
            for m, n, h in zip(marked["mark"], marked["number"], marked["horse_name"])) + "</div>")
    points = factor_view.model_factor_points(one) if "mark" in one else pd.DataFrame(index=one.index)
    one = pd.concat([one, points], axis=1)
    cols = {**({"mark": PRED_COLUMNS["mark"], "star": PRED_COLUMNS["star"]} if "mark" in one else {}), **ENTRY_COLUMNS,
            **(PRED_COLUMNS if prediction is not None else {})}
    # 要素ごとの勝率は、点数(0〜10点)で見せる。勝率そのものは、能力・オッズ込みの2列だけ残す
    for c, name in factor_view.MODEL_FACTORS.items():
        cols.pop(c, None)
        if f"pt_{c}" in one:
            cols[f"pt_{c}"] = st.column_config.NumberColumn(
                name, format="%.1f/10", help=f"{name}の要素だけで見た勝率を、このレースの中で0〜10点にした点数(最高の馬が10点)。"
                                          "8点以上に緑の色がつきます")
    if "factor_total" in one:
        # 総合スコアは馬名のすぐ右に置く
        at = list(cols).index("horse_name") + 1
        cols = {**dict(list(cols.items())[:at]), "factor_total": st.column_config.NumberColumn(
            "総合スコア", format="%.1f/100", help="各要素の点数の平均を100点満点にしたもの。モデルの勝率そのものではありません"),
            **dict(list(cols.items())[at:])}
    styled = one[[c for c in cols if c in one]].style.map(bracket_color, subset=["bracket"])
    if "mark" in one:
        styled = styled.map(mark_style, subset=["mark"])
    pt_cols = [c for c in cols if c.startswith("pt_")]
    if pt_cols:
        styled = styled.map(point_style, subset=pt_cols)
    if "factor_total" in cols:
        styled = styled.map(total_style, subset=["factor_total"])
    if prediction is None:
        st.caption("出馬表の見方", help=TIPS["entry_table"])
    else:
        st.caption("印と各列の見方", help=TIPS["race_table"])
    # 馬名の列までは、横にスクロールしても動かさない
    pinned = list(cols)[:list(cols).index("horse_name") + 1]
    st.dataframe(styled, hide_index=True, width="stretch",
                 column_config={k: ({**v, "pinned": True} if k in pinned else v) for k, v in cols.items()})
    values = _race_factor_values(conn, date, race_id) if date is not None and "mark" in one else None
    reviews = factor_view.race_reviews(one, points, values) if "mark" in one else []
    if reviews:
        st.subheader("予想の見解", anchor=False)
        st.caption("全頭について、要素ごとの点数と、その根拠(過去の成績など)を書いています。"
                   "印のある馬が先で、印のない馬は総合スコアの高い順です。点数はこのレース内の相対評価で、勝率そのものではありません。")
        for rv in reviews:
            title = f"{rv['mark']} {rv['number']} {rv['name']}".strip()
            with st.expander(f"{title} ― {rv['sub']}" if rv["sub"] else title, expanded=bool(rv["mark"])):
                st.html(review_html(rv))


def _race_factor_values(conn, date: dt.date, race_id: str | None):
    """このレースの各馬の要素の値(factor_values)。計算できなければ None(見解は点数だけになる)。

    race_id が None のときは計算だけして(保存と画面のメモリに残す)、何も返さない。
    """
    ped_bucket = conn.execute("SELECT COUNT(*) FROM pedigree").fetchone()[0] // 500
    stamp = f"{gui.entries_summary(conn, date.isoformat())['fetched_at']}/{ped_bucket}"
    try:
        with st.spinner("各要素の値を計算しています"):
            values = _factor_values(st.session_state["db_path"], date.isoformat(), stamp)
    except ValueError:
        return None
    return None if race_id is None else values[values["race_id"] == race_id]


def review_html(rv: dict) -> str:
    """1頭の見解。まず勝率と印の理由、その下に要素ごとの点数・評価・根拠を並べる。"""
    rows = "".join(
        f'<div class="kb-rv-row"><div class="kb-rv-name">{html.escape(f["name"])}</div>'
        f'<div class="kb-rv-pt {"kb-rv-hi" if f["pt"] >= factor_view.STRONG_PT else "kb-rv-lo" if f["pt"] <= factor_view.WEAK_PT else ""}">'
        + ("" if "fit" in rv else f'<div class="kb-rv-score"><b>{f["pt"]:.1f}</b><i>/10</i></div>') +
        f'<div class="kb-rv-grade">{html.escape(f["grade"])}</div></div>'
        f'<div class="kb-rv-text">{html.escape(f["text"])}</div></div>' for f in rv["rows"])
    return (f'<div class="kb-review"><p class="kb-rv-summary">{html.escape(rv["summary"])}</p>{rows}</div>')


def point_style(v) -> str:
    """要素の点数のセルの色。10点に近いほど濃い緑。"""
    if v != v or v is None or v < factor_view.STRONG_PT:
        return ""
    return ("background-color: rgba(46, 158, 91, 0.45); font-weight: 700" if v >= 9.95
            else "background-color: rgba(46, 158, 91, 0.2)")


def total_style(v) -> str:
    return "font-weight: 700; background-color: rgba(46, 158, 91, 0.12)"


MARK_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}


# ---------------------------------------------------------------- ファクター
STATE_ICON = {"ready": ":material/check_circle:", "partial": ":material/timelapse:", "missing": ":material/block:"}


@st.cache_data(show_spinner=False)
def _factor_values(db_path: str, date: str, stamp: str):
    return factor_view.factor_values_cached(db.connect(db_path), date)


def _factor_config(cols, frame=None) -> dict:
    """列の見出し・説明・書式。frame を渡すと、下の一覧にない数値の列も小数2桁で出す(54.200000 のようにしない)。"""
    out = {}
    for col, label in cols:
        info, good = factor_view.COLUMN_INFO.get(col, (None, None))
        help_ = f"{info}({factor_view.GOOD_LABEL[good]}。上位の馬に色がつきます)" if good else info
        if col.endswith("_rate") or col.endswith("_top3_rate"):
            out[col] = st.column_config.NumberColumn(label, format="percent", help=help_)
        elif col in ("early_pos_rel_3", "inner_pos", "weight_ratio", "l3f_rank_3", "track_front_bias", "track_inner_bias",
                     "jockey_roi", "pace_diff_3", "prev_pace_diff", "l3f_vs_lap_3", "time_diff_5", "avg_finish_5",
                     "bw_vs_avg", "weight_change", "win_odds"):
            out[col] = st.column_config.NumberColumn(label, format="%.2f", help=help_)
        elif col in ("jockey_rank", "jockey_wins_year", "pair_n", "tj_n", "n_front_runners", "number", "bracket", "age",
                     "popularity", "body_weight", "body_weight_diff", "days_since_last", "distance_change", "class_change",
                     "prev_pop_gap"):
            out[col] = st.column_config.NumberColumn(label, format="%d", help=help_)
        elif frame is not None and col in frame and pd.api.types.is_numeric_dtype(frame[col]):
            out[col] = st.column_config.NumberColumn(label, format="%.2f", help=help_)
        else:
            out[col] = st.column_config.TextColumn(label, help=help_)
    return out


OVERALL_COLUMNS = ["mark", "star", "p_win_market", "p_win_ability", "ev_market", "ev_ability"]
OVERALL_HIGH = {"p_win_market", "p_win_ability", "ev_market", "ev_ability", "score"}
SCORE_CONFIG = {
    "score_mark": st.column_config.TextColumn(
        "印(点数)", width="small",
        help="総合点の高い順に ◎ ○ ▲ △。モデルの勝率による印(右の「印」)とは別で、各ファクターの点数だけで決めています"),
    "score": st.column_config.NumberColumn(
        "総合点", format="%.1f", help="下の各列を、同じレースの中で0〜10点にして平均した100点満点の点数。人気・オッズは入れていません"),
}


def _factor_table(one, cols, extra=None):
    """各列の良い値の上位の馬に色をつける。1位は濃く、ほかは薄く。extra の列(予想の勝率・期待値)は馬名の右に入れる。"""
    df = one[[c for c, _ in cols]].copy()
    for c in df.columns:  # 文字の列の欠けは空欄にする(「None」と出さない)
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), "")
    if extra is not None:
        df = pd.concat([df.iloc[:, :2], extra, df.iloc[:, 2:]], axis=1)

    def paint(col):
        good = "high" if col.name in OVERALL_HIGH else factor_view.COLUMN_INFO.get(col.name, (None, None))[1]
        rank = factor_view.top_rank(col, good)
        return ["" if r != r else "background-color: rgba(46, 158, 91, 0.45); font-weight: 700" if r == 1
                else "background-color: rgba(46, 158, 91, 0.2)" for r in rank]
    return df.style.apply(paint)


def _factor_legend(cols) -> str:
    """各列の見方の一覧(ヘルプページ用)。"""
    items = []
    for col, label in cols:
        info, good = factor_view.COLUMN_INFO.get(col, (None, None))
        if not info:
            continue
        tag = (f'<span class="kb-fl-good">{factor_view.GOOD_LABEL[good]}</span>' if good
               else '<span class="kb-fl-none">良し悪しなし</span>')
        items.append(f"<li><b>{html.escape(label)}</b>{tag}<span>{html.escape(info)}</span></li>")
    return '<ul class="kb-factor-legend">' + "".join(items) + "</ul>"


def page_factors() -> None:
    header("Factors", "ファクター",
           "出走馬ごとの予想ファクターを、数字や成績のまま見られます。どれも過去の結果だけから作った値です。")
    conn = connection()
    status = factor_view.section_status(conn)
    st.html('<div class="kb-factor-status">' + "".join(
        f'<div class="kb-fs kb-fs-{s["state"]}"><b>{html.escape(s["name"])}</b><span>{html.escape(s["text"])}</span></div>'
        for s in status) + "</div>")

    today = dt.date.today()
    with st.container(border=True):
        date = st.date_input("対象日", gui.next_race_day(today), format="YYYY/MM/DD", key="factor_date")
        info = gui.entries_summary(conn, date.isoformat())
        if not info["races"]:
            st.caption(":material/info: この日の出馬表をまだ取得していません。「予想」ページで取得すると、ここで見られます。")
            return
        st.caption(f":material/check_circle: 取得済み {info['races']} レース・{info['horses']} 頭")
    races = gui.day_races(conn, date.isoformat())
    race, box = race_picker(races, key=f"factor_{date}")
    # 血統を取得している間も、500頭ふえるごとにだけ計算し直す(開くたびに全履歴の特徴量を作り直さないため)
    ped_bucket = conn.execute("SELECT COUNT(*) FROM pedigree").fetchone()[0] // 500
    if st.button("最新のデータで計算し直す", icon=":material/refresh:", help="血統などを取得したあと、すぐ反映したいときに押します"):
        _factor_values.clear()
    try:
        with st.spinner("ファクターを計算しています(同じ日は2回目から速くなります)"):
            values = _factor_values(st.session_state["db_path"], date.isoformat(), f"{info['fetched_at']}/{ped_bucket}")
    except ValueError as e:
        st.error(str(e))
        return
    one = values[values["race_id"] == race["race_id"]]
    with box:
        race_head(race)
        _show_race_flow(conn, race, one, date)
        _show_gate_bias(conn, race, date)
    tabs = st.tabs(["総合"] + [s["name"] for s in status])
    with tabs[0]:
        cols = [("number", "馬番"), ("horse_name", "馬名")] + factor_view.SUMMARY["cols"]
        cols = [c for c in cols if c[0] in one]
        cfg = {"number": st.column_config.NumberColumn("馬番", format="%d", pinned=True),
               "horse_name": st.column_config.TextColumn("馬名", pinned=True), **_factor_config(cols, one)}
        scored = factor_view.score_race(one)
        marked = scored[scored["score_mark"] != ""].sort_values("score", ascending=False)
        st.html('<div class="kb-marks">' + "".join(
            f'<span class="kb-mark"><b>{scored.loc[i, "score_mark"]}</b>{int(one.loc[i, "number"])} '
            f'{html.escape(str(one.loc[i, "horse_name"]))} {scored.loc[i, "score"]:.1f}</span>'
            for i in marked.index) + "</div>")
        st.caption("表の見方と総合点", help=TIPS["factor_overall"])
        extra = scored[["score_mark", "score"]]
        cfg.update(SCORE_CONFIG)
        pred = st.session_state.get("prediction")
        if pred is not None and pred[0] == date:
            got = pred[1].loc[pred[1]["race_id"] == race["race_id"], ["number", *OVERALL_COLUMNS]]
            got = got.dropna(axis=1, how="all")
            if len(got.columns) > 1:
                model = one[["number"]].merge(got, on="number", how="left").drop(columns=["number"])
                model.index = one.index
                extra = pd.concat([extra, model], axis=1)
                cfg.update({c: PRED_COLUMNS[c] for c in model.columns})
        st.dataframe(_factor_table(one, cols, extra), hide_index=True, width="stretch", column_config=cfg)
        with st.expander("点数の内訳", icon=":material/calculate:"):
            labels = dict(factor_view.score_columns())
            pts = scored[[c for c in scored if c.startswith("pt_")]].rename(columns=lambda c: labels.get(c[3:], c))
            st.dataframe(pd.concat([one[["number", "horse_name"]].rename(columns={"number": "馬番", "horse_name": "馬名"}),
                                    pts.round(1), scored[["score"]].rename(columns={"score": "総合点"})], axis=1)
                         .sort_values("総合点", ascending=False), hide_index=True, width="stretch")
    for i, (tab, section, st_) in enumerate(zip(tabs[1:], factor_view.SECTIONS, status)):
        with tab:
            if st_["state"] == "missing":
                st.warning(st_["text"], icon=":material/block:")
            elif st_["state"] == "partial":
                st.info(st_["text"], icon=":material/timelapse:")
            if not section["cols"]:
                continue
            cols = [("number", "馬番"), ("horse_name", "馬名")] + [c for c in section["cols"] if c[0] not in ("number", "jockey")]
            if section["name"] == "騎手・厩舎":
                cols.insert(2, ("jockey", "騎手"))
            cols = list(dict.fromkeys(c for c in cols if c[0] in one))
            cfg = {"number": st.column_config.NumberColumn("馬番", format="%d", pinned=True),
               "horse_name": st.column_config.TextColumn("馬名", pinned=True), **_factor_config(cols, one)}
            st.caption("表の見方", help=section["note"] + TIPS["factor_section"] + gui_help.link(f"factors-{i}"))
            st.dataframe(_factor_table(one, cols), hide_index=True, width="stretch", column_config=cfg)


# ---------------------------------------------------------------- 過去レース
def page_past_races() -> None:
    header("Past races", "過去レース",
           "出走馬ごとに過去走の評価をまとめ、そこから今回のレースでどう評価できるかの見込みを示します。"
           "どれも対象日より前の結果だけを使っています。")
    conn = connection()
    today = dt.date.today()
    with st.container(border=True):
        date = st.date_input("対象日", gui.next_race_day(today), format="YYYY/MM/DD", key="past_date")
        info = gui.entries_summary(conn, date.isoformat())
        if not info["races"]:
            st.caption(":material/info: この日の出馬表をまだ取得していません。「予想」ページで取得すると、ここで見られます。")
            return
        st.caption(f":material/check_circle: 取得済み {info['races']} レース・{info['horses']} 頭")
    races = gui.day_races(conn, date.isoformat())
    race, box = race_picker(races, key=f"past_{date}")
    with box:
        race_head(race)
    limit = st.select_slider("さかのぼる走数", [3, 5, 10, 20], value=5, key="past_limit")
    horses = past_races.horse_past_races(conn, race["race_id"], limit)
    ranked = sorted(horses, key=lambda h: (h["rank"] is None, h["rank"] or 0, h["number"]))
    st.html('<div class="kb-marks">' + "".join(
        f'<span class="kb-mark"><b>{h["rank"]}位</b>{h["number"]} {html.escape(str(h["horse_name"]))} {h["rating"]:.0f}点</span>'
        for h in ranked if h["rank"] is not None and h["rank"] <= 5) + "</div>")
    st.caption("今回の見込みの決め方", help=TIPS["past_eval"])
    outlook_cfg = {c: st.column_config.TextColumn(c, help=t) for c, t in gui_help.PAST_OUTLOOK_COLUMNS.items()}
    outlook_cfg["評価点"] = st.column_config.NumberColumn("評価点", help=gui_help.PAST_OUTLOOK_COLUMNS["評価点"], format="%.0f")
    st.dataframe(pd.DataFrame([{
        "順位": h["rank"], "馬番": h["number"], "馬名": h["horse_name"], "今回の見込み": h["outlook"],
        "評価点": h["rating"], "信頼度": h["confidence"], "対戦成績": h["h2h"],
        "上積み": "、".join(h["plus"]), "不安": "、".join(h["minus"])} for h in ranked]),
        hide_index=True, width="stretch", column_config=outlook_cfg)
    past_cfg = {c: st.column_config.TextColumn(c, help=t) for c, t in gui_help.PAST_COLUMNS.items()}
    past_cfg["評価"] = st.column_config.NumberColumn("評価", help=gui_help.PAST_COLUMNS["評価"], format="%.0f")
    past_cfg["今回への重み"] = st.column_config.NumberColumn("今回への重み", help=gui_help.PAST_COLUMNS["今回への重み"],
                                                       format="percent")
    for h in horses:
        runs = h["runs"]
        label = f"{h['number']} {h['horse_name']} ― {h['outlook']}"
        if h["rank"] is not None:
            label += f"({h['rank']}位・{h['rating']:.0f}点)"
        with st.expander(label, expanded=False):
            st.markdown(h["verdict"])
            if runs.empty:
                continue
            st.dataframe(runs[["日付", "レース", "条件", "結果", "上がり順", "人気", "評価", "kind", "重み"]]
                         .rename(columns={"kind": "今回との近さ", "重み": "今回への重み"}),
                         hide_index=True, width="stretch", column_config=past_cfg)
            for _, r in runs.iterrows():
                st.markdown(f"- **{r['日付']} {r['レース']}**({r['kind']}) {r['見解']}")


# ---------------------------------------------------------------- 買い方(1レース)
ALLOCATIONS = ["ケリー基準で配分", "配分なし(同額)"]
KELLY_CHOICES = {"半分(おすすめ)": 0.5, "4分の1(控えめ)": 0.25, "全部(攻めすぎ注意)": 1.0}
BET_MODES = ["自分の印から", "AIにおまかせ"]


def page_bets() -> None:
    header("Bets", "買い方",
           "出馬表に自分で印を付けると、最新のオッズから、おすすめの券種・買い目と資金の配分を出します。"
           "AI の予想だけで期待値の高い買い目を出すこともできます。")
    conn = connection()
    today = dt.date.today()
    pred = st.session_state.get("prediction")
    date = st.date_input("対象日", pred[0] if pred else gui.next_race_day(today), format="YYYY/MM/DD", key="bets_date")
    mode = st.segmented_control("出し方", BET_MODES, default=BET_MODES[0], key="bets_mode", width="stretch",
                                help="自分の印から: 付けた印で買い目を組み、資金の配分を出す。"
                                     "AIにおまかせ: 予想の勝率から、全券種で期待値の高い買い目を選ぶ") or BET_MODES[0]
    if mode == BET_MODES[0]:
        bets_from_marks(conn, date, pred[1] if pred and pred[0] == date else None)
    else:
        bets_auto(conn, date, pred)


def bets_from_marks(conn, date: dt.date, table) -> None:
    """出馬表に付けた印から、券種ごとの買い目と配分を出す。オッズはボタンを押したときに最新のものを取る。"""
    races = gui.day_races(conn, date.isoformat())
    if races.empty:
        with st.container(border=True):
            st.markdown("#### この日の出馬表がありません")
            st.caption("「予想」ページで、この日の出馬表を取得してから戻ってください。")
        return
    race, box = race_picker(races, key=f"marks_{date}")
    rid = race["race_id"]
    with box:
        race_head(race)
        marks = mark_editor(conn, rid, table)
        budget = st.number_input("このレースに使う金額(円)", 100, 1_000_000, 3_000, 100, key="marks_budget",
                                 help=TIPS["mark_alloc"])
    if list(marks.values()).count("◎") != 1:
        st.info("◎ を1頭付けると、買い目を出せます。○▲は2着候補、△☆は3着までの相手です(△☆は何頭でも)。",
                icon=":material/edit:")
        return

    st.subheader("最新オッズで計算", anchor=False)
    with st.container(border=True):
        store = st.session_state.setdefault("mark_odds", {})
        got = store.get(rid)
        label = "最新オッズを取り直して計算" if got else "最新オッズを取得して計算"
        if st.button(label, type="primary", icon=":material/refresh:"):
            try:
                with st.spinner("最新のオッズを取得しています(全券種。30秒ほど)"):
                    odds = gui.fetch_bet_odds(Fetcher(cache_dir=st.session_state["cache_dir"]), rid, conn=conn)
                if table is not None and rid in set(table["race_id"]):
                    blends = st.session_state.setdefault("blends", {})
                    if date not in blends:
                        with st.spinner("モデルとオッズの混ぜ方を決めています(初回だけ数分かかります)"):
                            blends[date] = gui.blend_for(conn, date)
                store[rid] = got = (odds, dt.datetime.now())
            except (ValueError, RuntimeError) as e:
                st.error(str(e))
        if not got:
            st.caption(":material/info: ボタンを押すと、このレースの全券種のオッズを取ってから計算します。"
                       "PC では JRA-VAN の速報オッズ、それ以外では netkeiba から取ります。")
            return
        odds, at = got
        blend = st.session_state.get("blends", {}).get(date)
        try:
            if table is not None and blend is not None and rid in set(table["race_id"]):
                probs = gui.mixed_probs(table, rid, odds, blend)
                source = "予想の勝率(オッズ込みのモデル)と、いまのオッズの確率を混ぜた勝率"
            else:
                probs = gui.market_probs(odds)
                source = "単勝オッズから出した勝率(この日を予想すると、モデルの勝率も混ぜます)"
            sums, prob = betting.mark_summaries([int(n) for n in probs["number"]], probs["p_win"].to_numpy(), odds,
                                                marks)
        except ValueError as e:
            st.warning(str(e), icon=":material/block:")
            return
        st.caption(f":material/schedule: オッズは {at:%H:%M} に取得しました。確率は{source}です。")
    if not sums:
        st.warning("印の買い目にオッズがありません(発売前の可能性があります)。", icon=":material/block:")
        return

    st.subheader("券種ごとの比較", anchor=False, help=TIPS["mark_kinds"])
    best = betting.recommend(sums)
    kinds = pd.DataFrame([{"おすすめ": "★" if i == best else "", "券種": s.kind, "点数": len(s.bets),
                           "的中確率": s.hit_prob, "合成オッズ": s.synthetic, "期待回収率": s.expected_return,
                           "外した点数": s.dropped, "金額": budget if i == best else 0} for i, s in enumerate(sums)])
    st.caption(f":material/edit: 「金額」の列に、券種ごとに買う金額を入れてください。はじめは、おすすめの券種(★)に{budget:,}円を入れています。")
    # 使う金額や券種が変わったら、金額の列をはじめの値に戻す
    edited = st.data_editor(kinds, hide_index=True, width="stretch", key=f"marks_kinds_{rid}_{budget}_{len(sums)}",
                            disabled=[c for c in kinds.columns if c != "金額"], column_config=KIND_COLUMNS)
    amounts = [int(a) if a == a and a is not None else 0 for a in edited["金額"]]
    chosen = [(s, a) for s, a in zip(sums, amounts) if a >= betting.UNIT]
    entered = sum(a for _, a in chosen)
    if entered > budget:
        st.warning(f"券種ごとの金額の合計({entered:,}円)が、このレースに使う金額({budget:,}円)を超えています。",
                   icon=":material/warning:")
    if not chosen:
        st.info("金額を入れた券種がありません(100円以上)。", icon=":material/edit:")
        return
    plan = betting.equal_payout_plan([s for s, _ in chosen], prob, budget, [a for _, a in chosen])
    st.subheader("買い目と金額", anchor=False)
    show_mark_plan(plan)


NO_MARK = "－"  # 印なし(空欄だと表に None と出るため)


def mark_editor(conn, race_id: str, table) -> dict[int, str]:
    """出馬表の印の欄を編集する。変えた印はすぐ DB に保存する。{馬番: 印} を返す。"""
    one = gui.race_entries(conn, race_id, table)
    saved = gui.load_user_marks(conn, race_id)
    ai = {int(n): m for n, m in zip(one["number"], one["mark"]) if m} if "mark" in one else {}
    df = pd.DataFrame({"mark": [saved.get(int(n), NO_MARK) for n in one["number"]]})
    df = pd.concat([df, one[[c for c in ["number", "bracket", "horse_name", "jockey", "win_odds", "popularity"]
                             if c in one]].reset_index(drop=True)], axis=1)
    if ai:
        df["ai_mark"] = [ai.get(int(n), "") for n in one["number"]]
    key = f"mark_edit_{race_id}"
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("AI の印を入れる", icon=":material/content_copy:", disabled=not ai, key=f"ai_{race_id}",
                 help="「予想」ページで予想した印を、自分の印としてそのまま入れます" if ai else "この日を予想すると使えます"):
        gui.save_user_marks(conn, race_id, ai)
        st.session_state.pop(key, None)
        st.rerun()
    if c2.button("印を消す", icon=":material/backspace:", disabled=not saved, key=f"clear_{race_id}"):
        gui.save_user_marks(conn, race_id, {})
        st.session_state.pop(key, None)
        st.rerun()
    edited = st.data_editor(
        df.style.map(bracket_color, subset=["bracket"]) if "bracket" in df else df, key=key, hide_index=True,
        width="stretch", height=35 * (len(df) + 1) + 3, disabled=[c for c in df.columns if c != "mark"],
        column_config={
            "mark": st.column_config.SelectboxColumn("印", options=[NO_MARK, *betting.MARKS], width="small",
                                                     required=True, help=TIPS["user_marks"]),
            **{k: v for k, v in ENTRY_COLUMNS.items() if k in df},
            "ai_mark": st.column_config.TextColumn("AIの印", width="small", help="「予想」ページの印(参考)")})
    marks = {int(n): m for n, m in zip(edited["number"], edited["mark"]) if m in betting.MARKS}
    if marks != saved:
        gui.save_user_marks(conn, race_id, marks)
    if marks:
        order = sorted(marks.items(), key=lambda x: (betting.MARKS.index(x[1]), x[0]))
        names = dict(zip(one["number"].astype(int), one["horse_name"]))
        st.html('<div class="kb-marks">' + "".join(
            f'<span class="kb-mark"><b>{m}</b>{n} {html.escape(str(names.get(n, "")))}</span>' for n, m in order) + "</div>")
    return marks


def show_mark_plan(plan) -> None:
    if not plan.bets:
        st.warning(plan.note, icon=":material/block:")
        if plan.top_ev:
            st.caption("参考: 期待値の高かった印の買い目")
            st.dataframe(gui.plan_table(plan.top_ev).drop(columns=["金額", "的中時の払戻"]), hide_index=True,
                         width="stretch", column_config=BET_COLUMNS)
        return
    pays = [b.stake * b.odds for b in plan.bets]
    pct = lambda x: f"{x:.0%}"
    stats = [("合計金額", f"{plan.total:,}円"), ("どれか的中", pct(plan.hit_prob)),
             ("当たったときの払戻", f"{min(pays):,.0f}〜{max(pays):,.0f}円"), ("期待回収率", pct(plan.expected_return))]
    st.html('<div class="kb-stats">' + "".join(
        f'<div><div class="kb-stat-label">{k}</div><div class="kb-stat-value">{v}</div></div>' for k, v in stats) + "</div>")
    if plan.note:
        st.caption(f":material/info: {plan.note}")
    if min(pays) < plan.total:
        st.caption(":material/warning: 払戻が合計金額に届かない買い目があります(当たってもトリガミ)。")
    st.dataframe(gui.plan_table(plan.bets), hide_index=True, width="stretch", column_config=BET_COLUMNS)


KIND_COLUMNS = {
    "おすすめ": st.column_config.TextColumn("", width="small",
                                         help="的中確率10%以上の券種のうち、期待回収率がいちばん高いもの"),
    "券種": st.column_config.TextColumn("券種"),
    "点数": st.column_config.NumberColumn("点数", format="%d", help="印から組んだ買い目のうち、オッズのある点数"),
    "的中確率": st.column_config.NumberColumn("的中確率", format="percent", help="どれか1つでも当たる確率"),
    "合成オッズ": st.column_config.NumberColumn("合成オッズ", format="%.2f",
                                           help="どれが当たっても同じ払戻になるように買ったときの、払戻 ÷ 合計金額。1未満はトリガミ"),
    "期待回収率": st.column_config.NumberColumn("期待回収率", format="percent",
                                           help="合成オッズ × 的中確率。確率が正しいときの、払戻の期待値 ÷ 合計金額"),
    "外した点数": st.column_config.NumberColumn("外した点数", format="%d", help="トリガミを避けるために外した、オッズの低い買い目の数"),
    "金額": st.column_config.NumberColumn("金額(円)", min_value=0, max_value=1_000_000, step=100, format="%d",
                                        help="この券種に使う金額。0なら買わない。どれが当たっても同じ払戻になるように、100円単位で配分します"),
}


def bets_auto(conn, date: dt.date, pred) -> None:
    """AI の予想の勝率から、全券種で期待値の高い買い目を選ぶ(これまでの買い方)。"""
    if pred is None or pred[0] != date:
        info = gui.entries_summary(conn, date.isoformat())
        with st.container(border=True):
            if not info["races"]:
                st.markdown("#### この日の出馬表がありません")
                st.caption("「予想」ページで、この日の出馬表とオッズを取得してから戻ってください。")
                return
            st.caption(f":material/info: 取得済み {info['races']} レース・{info['horses']} 頭。まだ予想していません。")
            if st.button("この日を予想する", type="primary", icon=":material/insights:"):
                try:
                    with st.spinner("学習して予想しています(数分かかります)"):
                        st.session_state["prediction"] = (date, gui.add_marks(gui.predict_table(conn, date)))
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        return
    date, table = pred
    races = gui.day_races(conn, date.isoformat())
    races = races[races["race_id"].isin(set(table["race_id"]))]
    if races.empty:
        st.info("予想のあるレースがありません。")
        return
    single = st.segmented_control("範囲", SCOPES, default=SCOPES[0], key="bets_scope", width="stretch") == SCOPES[1]
    race = None
    if single:
        race, box = race_picker(races, key=f"bets_{date}")
        with box:
            race_head(race)
    c1, c2, c3 = st.columns(3)
    mode = st.segmented_control("資金配分", ALLOCATIONS, default=ALLOCATIONS[0], key="bets_alloc", width="stretch",
                                help="ケリー基準は、期待値の高い買い目ほど多く張る配分。配分なしは、期待値の高い順に、どの買い目も同じ金額ずつ買う") or ALLOCATIONS[0]
    equal = mode == ALLOCATIONS[1]
    bankroll = c1.number_input("使える資金(円)", 100, 10_000_000, 100_000, 100,
                               help="配分を決める元になる資金です。全レースのときは、1日で使う合計の上限になります")
    cap = c2.number_input("1レースの上限(円)", 100, 1_000_000, 10_000, 100,
                          help="1つのレースに使う金額の上限。全レースのときは、残りの資金をレース数で割った額も超えません")
    if equal:
        stake = c3.number_input("1点の金額(円)", 100, 1_000_000, 100, 100,
                                help="どの買い目にも同じ金額を張る。上限まで、期待値の高い買い目から買う")
        kelly = 0.5
    else:
        stake = 0
        kelly = KELLY_CHOICES[c3.selectbox("賭け方の強さ", list(KELLY_CHOICES),
                                           help="資金が長い目で最も増える配分(ケリー基準)を、どれだけ抑えるか")]
    n = 1 if single else len(races)
    st.caption(f":material/info: 1レースごとに全券種のオッズを取るので、6回アクセスします(30秒ほど)。"
               f"{'' if single else f'全{n}レースで約{n * 30 // 60 + 1}分かかります。'}初回は、混ぜ方を決めるのに数分かかります。")
    if st.button("買い方を出す" if single else "全レースの買い方を出す", type="primary", icon=":material/calculate:"):
        try:
            blends = st.session_state.setdefault("blends", {})
            if date not in blends:
                with st.spinner("モデルとオッズの混ぜ方を決めています(数分かかります)"):
                    blends[date] = gui.blend_for(conn, date)
            targets = races if single is False else races[races["race_id"] == race["race_id"]]
            plans = plan_races(table, targets, blends[date], bankroll, cap, kelly, stake)
            st.session_state["bet_plans"] = (date, single, plans, blends[date])
        except (ValueError, RuntimeError) as e:
            st.error(str(e))
    saved = st.session_state.get("bet_plans")
    if not saved or saved[0] != date or saved[1] != single:
        return
    _, _, plans, blend = saved
    if single:
        if race["race_id"] not in plans:
            return
        show_plan(plans[race["race_id"]], blend)
        return
    show_day_plans(races, plans, blend)


def plan_races(table, targets, blend, bankroll: int, cap: int, kelly: float, equal_stake: int = 0) -> dict:
    """レースごとの買い方を出す。{race_id: (買い方 または 理由の文字列, レース名)}。

    複数レースのときは、残りの資金をレース数で割った額を1レースの上限にして、合計が資金を超えないようにする。
    """
    out = {}
    remaining = bankroll
    bar = st.progress(0.0, text="オッズを取得しています")
    fetcher = Fetcher(cache_dir=st.session_state["cache_dir"])
    rows = list(targets.itertuples())
    for i, r in enumerate(rows):
        label = f"{r.venue}{int(r.round)}R"
        bar.progress(i / len(rows), text=f"{label} のオッズを取得しています({i + 1}/{len(rows)})")
        if remaining < betting.UNIT:
            out[r.race_id] = ("資金を使い切りました", label)
            continue
        share = max(betting.UNIT, remaining // (len(rows) - i) // betting.UNIT * betting.UNIT)
        try:
            odds = gui.fetch_bet_odds(fetcher, r.race_id, conn=connection())
            plan, _ = gui.bet_plan_for(table, r.race_id, odds, blend, bankroll, min(cap, share, remaining), kelly,
                                     equal_stake)
        except (ValueError, RuntimeError) as e:
            out[r.race_id] = (str(e), label)
            continue
        out[r.race_id] = (plan, label)
        remaining -= plan.total
    bar.empty()
    return out


def show_day_plans(races, plans: dict, blend) -> None:
    rows, total, expected = [], 0, 0.0
    for r in races.itertuples():
        got = plans.get(r.race_id)
        if got is None:
            continue
        plan, label = got
        if isinstance(plan, str):
            rows.append({"レース": label, "結果": plan, "合計金額": 0, "どれか的中": None, "期待回収率": None})
        elif not plan.bets:
            rows.append({"レース": label, "結果": "見送り", "合計金額": 0, "どれか的中": None, "期待回収率": None})
        else:
            total += plan.total
            expected += plan.expected_profit
            rows.append({"レース": label, "結果": f"{len(plan.bets)}点", "合計金額": plan.total,
                         "どれか的中": plan.hit_prob, "期待回収率": plan.expected_return})
    stats = [("買うレース", f"{sum(1 for r in rows if r['合計金額'])}/{len(rows)}"), ("合計金額", f"{total:,}円"),
             ("期待収支", f"{expected:+,.0f}円")]
    st.html('<div class="kb-stats">' + "".join(
        f'<div><div class="kb-stat-label">{k}</div><div class="kb-stat-value">{v}</div></div>' for k, v in stats) + "</div>")
    st.caption("この表の見方", help=TIPS["day_plans"])
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "レース": st.column_config.TextColumn("レース", help="競馬場とレース番号"),
        "結果": st.column_config.TextColumn("結果", help="買う点数。見送りは、期待値が1を超え的中確率が3%以上の買い目がなかったレース"),
        "合計金額": st.column_config.NumberColumn("合計金額(円)", format="%d", help="そのレースで買う金額の合計"),
        "どれか的中": st.column_config.NumberColumn("どれか的中", format="percent", help="買った買い目のどれか1つでも当たる確率"),
        "期待回収率": st.column_config.NumberColumn("期待回収率", format="percent", help="予想が当たっている前提の、払戻の期待値 ÷ 買った金額")})
    for r in races.itertuples():
        got = plans.get(r.race_id)
        if got is None or isinstance(got[0], str) or not got[0].bets:
            continue
        with st.expander(f"{got[1]}  {gui.text_or_blank(r.name)}  {got[0].total:,}円"):
            show_plan(got, blend)


def show_plan(got, blend) -> None:
    plan, _ = got
    if isinstance(plan, str):
        st.warning(plan, icon=":material/block:")
        return
    if not plan.bets:
        st.warning(plan.note, icon=":material/block:")
        if plan.top_ev:
            st.caption("参考: 期待値の高かった買い目(どれも的中確率3%未満か、期待値1未満)")
            st.dataframe(gui.plan_table(plan.top_ev).drop(columns=["金額", "的中時の払戻"]), hide_index=True,
                         width="stretch", column_config=BET_COLUMNS)
        return
    pct = lambda x: f"{x:.0%}"
    stats = [("合計金額", f"{plan.total:,}円"), ("どれか的中", pct(plan.hit_prob)),
             ("期待回収率", pct(plan.expected_return)), ("期待収支", f"{plan.expected_profit:+,.0f}円"),
             ("全部外れたとき", f"{plan.worst:,}円")]
    st.html('<div class="kb-stats">' + "".join(
        f'<div><div class="kb-stat-label">{k}</div><div class="kb-stat-value">{v}</div></div>' for k, v in stats) + "</div>")
    if plan.note:
        st.caption(f":material/info: {plan.note}")
    st.caption("期待値と確率の出し方", help=TIPS["bet_plan"].format(a=blend.a, b=blend.b))
    st.dataframe(gui.plan_table(plan.bets), hide_index=True, width="stretch", column_config=BET_COLUMNS)


BET_COLUMNS = {
    "券種": st.column_config.TextColumn("券種", help="馬券の種類(単勝・複勝・馬連・ワイド・馬単・三連複・三連単)"),
    "買い目": st.column_config.TextColumn("買い目", help="買う馬番の組み合わせ。馬単・三連単は着順どおり、他は順不同"),
    "オッズ": st.column_config.NumberColumn("オッズ", format="%.1f",
                                         help="当日のオッズ(倍)。複勝とワイドは幅のある下限で計算"),
    "的中確率": st.column_config.NumberColumn("的中確率", format="percent",
                                          help="この買い目が当たる確率。混ぜた勝率から、1〜3着の組み合わせをハーヴィルの式で出したもの"),
    "期待値": st.column_config.NumberColumn("期待値", format="%.2f",
                                         help="的中確率 × オッズ。1を超えると、確率が正しければ長い目で得になる目安"),
    "金額": st.column_config.NumberColumn("金額(円)", format="%d", help="この買い目に張る金額(100円単位)"),
    "的中時の払戻": st.column_config.NumberColumn("的中時の払戻(円)", format="%d", help="当たったときの払い戻し(金額 × オッズ)"),
}


# ---------------------------------------------------------------- 回顧
def page_review() -> None:
    from keibaai import weekly

    header("Review", "回顧", "発走前に保存した予想と、レースの結果を照らし合わせます。毎週の処理(keibaai weekly)で予想が溜まります。")
    conn = connection()
    days = gui.review_dates(conn)
    if not days:
        with st.container(border=True):
            st.markdown("#### 保存された予想がまだありません")
            st.caption("PC で `keibaai weekly` を実行すると、今週末の予想が保存されます。")
        return
    day = st.selectbox("開催日", days)
    r = weekly.review_day(conn, day)
    if not r.summary["races"]:
        st.info("この日の結果はまだ取得されていません。取得したあとに表示されます。", icon=":material/hourglass_empty:")
        return
    s = r.summary
    pct = lambda x: "—" if x != x else f"{x:.0f}%"
    stats = [("予想したレース", f"{s['races']}"), ("◎が1着", f"{s['honmei_wins']}"), ("◎の回収率", pct(s["honmei_roi"])),
             ("期待値≧1の回収率", pct(s["ev_roi"]))]
    st.html('<div class="kb-stats">' + "".join(
        f'<div><div class="kb-stat-label">{k}</div><div class="kb-stat-value">{v}</div></div>' for k, v in stats) + "</div>")
    st.caption("数字の見方", help=TIPS["review"])
    scope = st.segmented_control("印ごとの成績の範囲", ["この日", "保存済みの全期間"], default="この日", key="review_scope") or "この日"
    marks = r.marks
    if scope != "この日":  # 1日だと頭数が少ないので、結果の出ている全部の日をまとめて出す
        marks = weekly.mark_stats(pd.concat([weekly.review_day(conn, d).horses for d in days], ignore_index=True))
    if marks is not None and not marks.empty:
        st.subheader("印ごとの成績", anchor=False)
        pct = lambda label, t: st.column_config.NumberColumn(label, format="percent", help=t)
        st.dataframe(marks.drop(columns=["top3"]).style.map(mark_style, subset=["mark"]), hide_index=True,
                     width="stretch", column_config={
            "mark": st.column_config.TextColumn("印", width="small"),
            "n": st.column_config.NumberColumn("頭数", format="%d", help="その印を付けて、走った馬の数(取消・除外は除く)"),
            "wins": st.column_config.NumberColumn("1着", format="%d", help="勝った頭数"),
            "win_rate": pct("勝率", "1着の割合"), "top2_rate": pct("連対率", "2着以内の割合"),
            "top3_rate": pct("馬券内率", "3着以内の割合"),
            "roi": pct("単勝回収率", "その印の馬の単勝を100円ずつ買った場合の、払戻 ÷ 投資(確定オッズ)")})

    st.subheader("レースごと", anchor=False)
    races = r.races.sort_values(["venue", "round"]).reset_index(drop=True)
    race, box = race_picker(races, key=f"review_{day}")
    heads = gui.day_races(conn, day).set_index("race_id")
    with box:
        if race["race_id"] in heads.index:
            race_head(heads.loc[race["race_id"]])
        else:
            st.markdown(f"#### {race['venue']}{int(race['round'])}R {gui.text_or_blank(race['race_name'])}")
        review_race(r.horses[r.horses["race_id"] == race["race_id"]], race,
                    (r.reviews or {}).get(race["race_id"]))


def review_race(g, race=None, reviews=None) -> None:
    """1レースの回顧: 印を付けた馬すべての着順と、1〜3着の馬。"""
    from keibaai import weekly

    if race is not None and race.get("flow"):  # 発走前の展開予想と、結果から出した実際の展開
        if race.get("forecast"):
            st.caption(f"展開の予想: {race['forecast']}", help=gui_help.REVIEW_COLUMNS["flow"][1])
        st.caption(f"実際の展開: {race['flow']}")
        if race.get("track"):
            st.caption(f"馬場・枠: {race['track']}")

    def place_style(v) -> str:
        return "font-weight: 700; background-color: rgba(46, 158, 91, 0.2)" if v == v and v is not None and v <= 3 else ""

    g = g.assign(finish_text=[weekly.finish_text(f, s) for f, s in zip(g["finish"], g.get("status", [None] * len(g)))])
    cols = {"mark": PRED_COLUMNS["mark"], "number": PRED_COLUMNS["number"], "horse_name": PRED_COLUMNS["horse_name"],
            "finish_text": st.column_config.TextColumn("着順", help="取消・除外の馬は理由を出します"),
            "final_pop": st.column_config.NumberColumn("人気", format="%d", help="確定の単勝人気"),
            "odds": st.column_config.NumberColumn("単勝", format="%.1f", help="確定の単勝オッズ(なければ予想時のもの)"),
            "p_win_ability": st.column_config.NumberColumn("勝率(能力)", format="percent", help="発走前に出した、オッズを見ない勝率")}
    marked = g[g["mark"].isin(weekly.MARK_ORDER)]
    marked = marked.assign(_o=marked["mark"].map(weekly.MARK_ORDER.index)).sort_values(["_o", "number"])
    st.caption("印を付けた馬の着順(3着以内に色をつけています)")
    st.dataframe(marked[list(cols)].style.map(mark_style, subset=["mark"]).apply(
        lambda c: [place_style(marked.at[i, "finish"]) for i in c.index], subset=["finish_text"]),
        hide_index=True, width="stretch", column_config=cols)
    top = g[g["finish"].between(1, 3)].sort_values("finish")
    if not top.empty:
        st.html('<div class="kb-marks">' + "".join(
            f'<span class="kb-mark"><b>{int(f)}着</b>{html.escape(m or "")}{int(n)} {html.escape(str(h))}'
            f'{"" if o != o or o is None else f"({o:.1f}倍)"}</span>'
            for f, m, n, h, o in zip(top["finish"], top["mark"], top["number"], top["horse_name"], top["odds"])) + "</div>")
    with st.expander("全馬の予想と結果", icon=":material/table:"):
        allh = g.sort_values(["finish", "number"], na_position="last")
        st.dataframe(allh[list(cols)].style.map(mark_style, subset=["mark"]), hide_index=True, width="stretch",
                     column_config=cols)
    if reviews and reviews["horses"]:
        st.subheader("回顧の見解", anchor=False)
        st.caption("そのレースの展開・枠・馬場状態・コース・距離に、各馬が向いていたかを、結果と過去の成績から機械的に判定しています。"
                   "印のある馬が先です。向いたは緑、向かなかったは赤みで示します。",
                   help=gui_help.REVIEW_COLUMNS["flow"][1])
        for rv in reviews["horses"]:
            title = f"{rv['mark']} {rv['number']} {rv['name']}".strip()
            with st.expander(f"{title} ― {rv['sub']}", expanded=bool(rv["mark"])):
                st.html(review_html(rv))


# ---------------------------------------------------------------- 買い方の検証
def page_backtest() -> None:
    header("Simulation", "馬券シミュレーション",
           "過去のレースで「こう買っていたら」を再現します。学習は開始日より前だけ、買うのは開始日以後のレースだけです。")
    conn = connection()
    first, last = gui.date_bounds(conn)
    if first is None:
        with st.container(border=True):
            st.markdown("#### まだデータがありません")
            st.page_link(PAGES["collect"], label="取得へ", icon=":material/arrow_forward:")
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
        start = c1.date_input("買う期間の開始日", max(first, last - dt.timedelta(days=90)), min_value=first,
                              max_value=last, format="YYYY/MM/DD")
        model = c2.selectbox("勝率のモデル", ["能力のみ", "オッズ込み", "オッズと混ぜる"],
                             help="「オッズと混ぜる」は、モデルの勝率と市場の確率を、開始日の前の1年で決めた割合で混ぜます")
        stake = c3.number_input("1点の金額(円)", 100, 100000, 100, step=100)
        c4, c5, c6 = st.columns([2, 2, 1], vertical_alignment="bottom")
        ev_min = c4.slider("期待値の基準", 0.5, 3.0, 1.0, 0.1, help="これ以上の馬だけ買うルールの基準です")
        cap = c5.number_input("オッズの上限(倍・0で無制限)", 0.0, 1000.0, 0.0, step=10.0,
                              help="大穴は期待値が大きく出やすいので、上限をつけて試せます")
        run = c6.button("再現する", type="primary", icon=":material/play_arrow:", width="stretch")
    if run:
        try:
            with st.spinner("学習して買い方を再現しています(数分かかります)"):
                st.session_state["backtest"] = (start, gui.backtest_run(
                    conn, start.isoformat(), model == "オッズ込み", ev_min, cap or None, int(stake),
                    blend=model == "オッズと混ぜる"))
        except ValueError as e:
            st.error(str(e))

    if "backtest" not in st.session_state:
        return
    when, (n_races, table, settled) = st.session_state["backtest"]
    st.subheader("結果", anchor=False, help=TIPS["backtest_result"])
    st.caption(f"{when:%Y/%m/%d} 以降の {n_races:,} レース。単勝のみ、確定オッズで計算しています。")
    best = table["roi"].max()
    for col, (_, r) in zip(st.container(key="rules").columns(len(table), gap="small"), table.iterrows()):
        with col.container(border=True, height="stretch"):
            badge = " :green-badge[回収率 最高]" if r["roi"] == best and r["n_bets"] else ""
            st.markdown(f"**{r['rule']}**{badge}")
            hit = "—" if r["hit_rate"] != r["hit_rate"] else f"{r['hit_rate']:.1%}"
            stats = [("回収率", f"{r['roi']:.1f}%" if r["n_bets"] else "—"), ("的中率", hit),
                     ("収支", f"{r['profit']:+,}円"), ("買い目数", f"{r['n_bets']:,}")]
            st.html('<div class="kb-stats">' + "".join(
                f'<div><div class="kb-stat-label">{k}</div><div class="kb-stat-value">{v}</div></div>'
                for k, v in stats) + "</div>")

    from keibaai.backtest import balance_by_date

    # 全頭買いは大きく負けて縦軸を占領するので、推移のグラフには出さない
    curves = pd.concat([balance_by_date(s).rename(name) for name, s in settled.items() if not name.startswith("全頭")], axis=1).sort_index().ffill().fillna(0)
    long = curves.reset_index().melt("date", var_name="ルール", value_name="収支(円)")
    st.altair_chart(alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y/%m")), y=alt.Y("収支(円):Q", title=None), color=alt.Color("ルール:N", title=None),
        tooltip=["date:T", "ルール:N", "収支(円):Q"]).properties(height=280), width="stretch")
    with st.expander("すべての指標", icon=":material/table:"):
        from keibaai.backtest import COLUMNS
        st.dataframe(table.rename(columns=COLUMNS), hide_index=True, width="stretch",
                     column_config={c: st.column_config.Column(help=t) for c, t in gui_help.BACKTEST_COLUMNS.items()})


# ---------------------------------------------------------------- ヘルプ
def _column_list(columns: dict) -> str:
    """st.column_config の辞書から、見出しと説明の一覧(_factor_legend と同じ見た目)を作る。"""
    items = [f"<li><b>{html.escape(c.get('label') or k)}</b><span>{html.escape(c['help'])}</span></li>"
             for k, c in columns.items() if isinstance(c, dict) and c.get("help")]
    return '<ul class="kb-factor-legend">' + "".join(items) + "</ul>"


def page_help() -> None:
    header("Help", "ヘルプ", "各ページの数字と表の見方です。画面の ? にマウスを乗せると、ここの要点が出ます。")
    with st.container(border=True):
        st.markdown("##### 目次")
        st.markdown(" ・ ".join(f"[{title}](#{anchor})" for anchor, title, _ in gui_help.HELP))
    for anchor, title, body in gui_help.HELP:
        st.subheader(title, anchor=anchor)
        st.markdown(body)
        if anchor == "predict":
            with st.expander("出馬表と予想の列", icon=":material/table:"):
                st.html(_column_list({**ENTRY_COLUMNS, **PRED_COLUMNS}))
        elif anchor == "bets":
            with st.expander("買い目の表の列", icon=":material/table:"):
                st.html(_column_list(BET_COLUMNS))
        elif anchor == "factors-score":  # 区分ごとの列の説明は、ファクターの表と同じ一覧から作る
            for i, section in enumerate(factor_view.SECTIONS):
                st.subheader(f"ファクター:{section['name']}", anchor=f"factors-{i}")
                st.markdown(section["note"])
                st.html(_factor_legend(section["cols"]))


# ---------------------------------------------------------------- 設定
def page_settings() -> None:
    header("Settings", "設定", "データの保存場所です。ふつうは変更不要です。")
    with st.container(border=True):
        # 入力欄の状態は画面を移ると消えるので、確定した値を別のキーに写しておく
        for key, label in [("db_path", "DB ファイル"), ("cache_dir", "保存済みページのフォルダ")]:
            st.text_input(label, st.session_state[key], key=f"_{key}",
                          on_change=lambda k=key: st.session_state.update({k: st.session_state[f"_{k}"]}))
        st.caption(f":material/folder: 作業フォルダ: `{Path.cwd()}`")


PAGES = {
    "overview": st.Page(page_overview, title="データ", icon=":material/dashboard:", default=True),
    "collect": st.Page(page_collect, title="取得", icon=":material/download:", url_path="collect"),
    "progress": st.Page(page_progress, title="進み具合", icon=":material/timelapse:", url_path="progress"),
    "reparse": st.Page(page_reparse, title="再解析", icon=":material/refresh:", url_path="reparse"),
    "predict": st.Page(page_predict, title="予想", icon=":material/sports_score:", url_path="predict"),
    "bets": st.Page(page_bets, title="買い方", icon=":material/calculate:", url_path="bets"),
    "factors": st.Page(page_factors, title="ファクター", icon=":material/tune:", url_path="factors"),
    "past": st.Page(page_past_races, title="過去レース", icon=":material/manage_search:", url_path="past"),
    "train": st.Page(page_train, title="学習・評価", icon=":material/insights:", url_path="train"),
    "review": st.Page(page_review, title="回顧", icon=":material/history:", url_path="review"),
    "backtest": st.Page(page_backtest, title="シミュレーション", icon=":material/payments:", url_path="backtest"),
    "settings": st.Page(page_settings, title="設定", icon=":material/settings:", url_path="settings"),
    "help": st.Page(page_help, title="ヘルプ", icon=":material/help:", url_path="help"),
}
st.navigation(list(PAGES.values()), position="top").run()
