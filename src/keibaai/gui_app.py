"""KeibaAI の操作画面(Streamlit)。`keibaai-gui` で起動する。"""
from __future__ import annotations

import contextlib
import datetime as dt
import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from keibaai import db, gui
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
    cols[0].metric("レース", f"{s['races']:,}", border=True, height="stretch")
    cols[1].metric("出走", f"{s['results']:,}", border=True, height="stretch")
    cols[2].metric("馬", f"{s['horses']:,}", border=True, height="stretch")
    first, last = dt.date.fromisoformat(s["first"]), dt.date.fromisoformat(s["last"])
    cols[3].metric("期間", f"{(last - first).days // 30 + 1} か月", border=True,
                   delta=f"{first:%Y/%m} – {last:%Y/%m}", delta_color="off", delta_arrow="off",
                   height="stretch")

    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        st.markdown("##### 月別レース数")
        st.altair_chart(bar_chart(monthly, "month", "races"), width="stretch")
    with right.container(border=True):
        st.markdown("##### 競馬場別")
        st.altair_chart(bar_chart(gui.races_by_venue(conn), "venue", "races", horizontal=True), width="stretch")

    with st.container(border=True):
        st.markdown("##### 最近のレース")
        st.dataframe(gui.recent_races(conn), hide_index=True, width="stretch", height=380, column_config={
            "date": st.column_config.DateColumn("日付", format="YYYY/MM/DD"), "venue": "競馬場",
            "round": st.column_config.NumberColumn("R", format="%dR"), "name": "レース名", "course_type": "芝/ダ",
            "distance": st.column_config.NumberColumn("距離", format="%dm"), "ground": "馬場", "runners": "頭数"})


# ---------------------------------------------------------------- データ取得
def _apply_preset() -> None:
    name = st.session_state.get("preset")
    if name:
        st.session_state["start"], st.session_state["end"] = gui.preset_range(name, dt.date.today())


def page_collect() -> None:
    header("Collect", "データ取得",
           "選んだ期間の土日・月曜の JRA レース結果を netkeiba から取得します。"
           "取得済みのレースは飛ばすので、同じ期間を何度実行しても大丈夫です。")
    today = dt.date.today()
    st.session_state.setdefault("start", gui.preset_range("先週末", today)[0])
    st.session_state.setdefault("end", gui.preset_range("先週末", today)[1])

    with st.container(border=True):
        st.segmented_control("よく使う期間", gui.PRESETS, key="preset", on_change=_apply_preset)
        c1, c2 = st.columns(2)
        start = c1.date_input("開始日", key="start", format="YYYY/MM/DD", max_value=today)
        end = c2.date_input("終了日", key="end", format="YYYY/MM/DD", max_value=today)
        days = gui.race_days(start, end) if start <= end else []
        if start > end:
            st.error("終了日が開始日より前になっています。")
        else:
            st.caption(f":material/event: 対象 {len(days)} 日 ・ :material/schedule: 最大で約 {len(days) * 2} 分"
                       "(サーバーに負荷をかけないよう 1 ページごとに 3 秒以上あけます)")
        run = st.button("取得を開始", type="primary", icon=":material/download:", disabled=not days, width="stretch")

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
                    total = gui.collect(conn, Fetcher(cache_dir=st.session_state["cache_dir"]), start, end, on_day)
            except Exception as e:  # 連続失敗(アクセス制限の可能性)や通信エラー。途中までの分は保存済み
                status.update(label="取得が止まりました", state="error")
                st.error(str(e))
            else:
                status.update(label=f"完了しました。新しく {total} レースを保存しました", state="complete")


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
        test_from = c1.date_input("評価開始日", max(first, last - dt.timedelta(days=90)), min_value=first,
                                  max_value=last, format="YYYY/MM/DD",
                                  help=f"取得済み期間は {first:%Y/%m/%d} – {last:%Y/%m/%d} です")
        run = c2.button("学習して評価", type="primary", icon=":material/model_training:", width="stretch")
    if run:
        try:
            with st.spinner("学習しています(データ量によって数分かかります)"):
                st.session_state["report"] = (test_from, gui.train_reports(conn, test_from.isoformat()))
        except ValueError as e:
            st.error(str(e))

    if "report" not in st.session_state:
        return
    when, report = st.session_state["report"]
    st.subheader("評価結果", anchor=False)
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
        st.dataframe(report, hide_index=True, width="stretch", column_config={
            "勝LogLoss": st.column_config.NumberColumn(format="%.4f"),
            "勝AUC": st.column_config.NumberColumn(format="%.3f"),
            "複AUC": st.column_config.NumberColumn(format="%.3f"),
            "1番手的中": st.column_config.NumberColumn(format="percent"),
            "単勝回収率(%)": st.column_config.NumberColumn(format="%.1f"),
        })
        st.caption("勝AUC は 0.5 が当てずっぽうで、高いほどよい指標です。単勝の控除率は約 20% なので、"
                   "回収率が 100% を超えるのは簡単ではありません。")


# ---------------------------------------------------------------- 予想
PRED_COLUMNS = {
    "mark": st.column_config.TextColumn("印", width="small"),
    "number": st.column_config.NumberColumn("馬番", format="%d"), "horse_name": "馬名", "jockey": "騎手",
    "popularity": st.column_config.NumberColumn("人気", format="%d"),
    "win_odds": st.column_config.NumberColumn("単勝", format="%.1f"),
    "p_win_ability": st.column_config.ProgressColumn("勝率(能力)", min_value=0, max_value=1, format="percent"),
    "p_win_market": st.column_config.ProgressColumn("勝率(オッズ込)", min_value=0, max_value=1, format="percent"),
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
    "ev_ability": st.column_config.NumberColumn("期待値(能力)", format="%.2f"),
    "ev_market": st.column_config.NumberColumn("期待値(オッズ込)", format="%.2f"),
}
ENTRY_COLUMNS = {
    "bracket": st.column_config.NumberColumn("枠", format="%d"),
    "number": PRED_COLUMNS["number"], "horse_name": "馬名", "sex_age": "性齢",
    "weight_carried": st.column_config.NumberColumn("斤量", format="%.1f"), "jockey": "騎手",
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
        single = st.segmented_control("範囲", SCOPES, default=SCOPES[0], key="pred_scope",
                                      width="stretch") == SCOPES[1]
        if not single:
            c1, c2 = st.columns(2)
            fetch = c1.button("全レースの出馬表とオッズを取得", icon=":material/download:", width="stretch")
            predict = c2.button("予想する", type="primary", icon=":material/insights:", width="stretch",
                                disabled=info["races"] == 0)

    picked = None
    if single:
        picked, fetch, predict = single_race_controls(conn, date)
        if picked is None:
            return
    st.caption(":material/info: 取得には1レースあたり数秒かかります。オッズは発売開始後に出ます。"
               "オッズが更新されたら、もう一度取得してください。")

    if fetch:
        fetch_entries(conn, date, None if picked is None else [picked["race_id"]])

    if predict:
        try:
            with st.spinner("学習して予想しています(数分かかります)"):
                st.session_state["prediction"] = (date, gui.add_marks(gui.predict_table(conn, date)))
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
                            picked["race_id"], table)
        return

    if table is not None:
        st.subheader("期待値の高い馬", anchor=False)
        key = "ev_market" if table["ev_market"].notna().any() else "ev_ability"
        top = table.dropna(subset=[key]).nlargest(8, key)
        with st.container(border=True):
            st.dataframe(top.assign(race=top["venue"] + top["round"].astype(int).astype(str) + "R")[
                ["race", "mark", "number", "horse_name", "win_odds", "p_win_ability", "p_win_market",
                 "ev_ability", "ev_market"]].style.map(mark_style, subset=["mark"]),
                hide_index=True, width="stretch", column_config={**PRED_COLUMNS, "race": "レース"})
            st.caption("期待値 = 勝率 × 単勝オッズ。1 を超えると、予測が正しければ長い目で得になる目安です。"
                       "過去のレースでの検証はこれからで、今は参考表示です。")

    races = gui.day_races(conn, date.isoformat())
    if not races.empty:
        st.subheader("出馬表" if table is None else "レース別", anchor=False)
        race, box = race_picker(races, key=f"board_{date}")
        with box:
            race_detail(conn, race, race["race_id"], table)


def single_race_controls(conn, date: dt.date):
    """1レースだけのとき: レースを選ぶ欄と、そのレースの取得・予想ボタン。選んだレースと押されたボタンを返す。"""
    listed = st.session_state.setdefault("race_lists", {}).get(date.isoformat(), [])
    choices = gui.race_choices(conn, date.isoformat(), listed)
    if choices.empty:
        with st.container(border=True):
            st.caption(":material/info: この日のレース一覧をまだ読み込んでいません。")
            if st.button("レース一覧を読み込む", icon=":material/list:", type="primary"):
                with st.spinner("レース一覧を読み込んでいます"):
                    try:
                        ids = Fetcher(cache_dir=st.session_state["cache_dir"]).race_ids_on(date.strftime("%Y%m%d"))
                    except Exception as e:
                        st.error(str(e))
                        return None, False, False
                if not ids:
                    st.warning("この日のレースは見つかりませんでした。")
                    return None, False, False
                st.session_state["race_lists"][date.isoformat()] = ids
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
    with st.status("取得しています…", expanded=True) as status:
        bar = st.progress(0.0)
        log = LogBox()

        def on_race(i, n, race_id):
            bar.progress(i / n if n else 1.0, f"{i + 1}/{n} レース")

        try:
            with contextlib.redirect_stdout(log):
                n = gui.fetch_entries_for(conn, Fetcher(cache_dir=st.session_state["cache_dir"]), date, on_race,
                                          race_ids)
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


def race_detail(conn, race, race_id: str, prediction) -> None:
    """レース名・条件と出馬表。予想があれば印・勝率・期待値も出す。"""
    meta = [f"{race.course_type or ''}{race.distance}m" if race.distance else None, race.ground,
            f"{race.start_time} 発走" if race.start_time else None, f"{race.runners}頭"]
    st.html(f'<div class="kb-race-head"><span class="kb-race-no">{int(race["round"])}R</span>'
            f'<span class="kb-race-name">{html.escape(race["name"] or "")}</span></div>'
            f'<div class="kb-race-meta">{html.escape(" ・ ".join(m for m in meta if m))}</div>')
    one = gui.race_entries(conn, race_id, prediction)
    if "mark" in one:
        marked = one[one["mark"] != ""].sort_values("mark", key=lambda m: m.map(MARK_ORDER.get))
        st.html('<div class="kb-marks">' + "".join(
            f'<span class="kb-mark"><b>{m}</b>{n} {html.escape(str(h))}</span>'
            for m, n, h in zip(marked["mark"], marked["number"], marked["horse_name"])) + "</div>")
    cols = {**({"mark": PRED_COLUMNS["mark"]} if "mark" in one else {}), **ENTRY_COLUMNS,
            **(PRED_COLUMNS if prediction is not None else {})}
    styled = one[[c for c in cols if c in one]].style.map(bracket_color, subset=["bracket"])
    if "mark" in one:
        styled = styled.map(mark_style, subset=["mark"])
    st.dataframe(styled, hide_index=True, width="stretch", column_config=cols)
    if prediction is None:
        st.caption(":material/info: 「予想する」を押すと、予想印と勝率・期待値の列が加わります。")
    else:
        st.caption("印は勝率の高い順に ◎ ○ ▲ △(オッズがあればオッズ込み、なければ能力のみの勝率)。"
                   "☆ は印のない馬のうち能力の期待値がいちばん高く、1 を超える穴馬です。")


MARK_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}


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
    st.markdown(weekly.render_markdown(r).replace("# ", "### ", 1))
    with st.expander("馬ごとの予想と結果", icon=":material/table:"):
        cols = ["venue", "round", "number", "horse_name", "mark", "p_win_ability", "ev_ability", "popularity", "finish"]
        st.dataframe(r.horses[cols].sort_values(["venue", "round", "number"]), hide_index=True, width="stretch")
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
    st.subheader("結果", anchor=False)
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
        st.dataframe(table.rename(columns=COLUMNS), hide_index=True, width="stretch")
    st.info("回収率の下限〜上限は、買い目を引き直して出した幅です。買い目が少ないほど広く、"
            "回収率が100%を超えても幅が100%をまたぐなら、たまたまの可能性が高いです。"
            "また、確定オッズは買う時点のオッズより低いことが多く、実際の成績は表より悪くなりがちです。", icon=":material/info:")


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
    "reparse": st.Page(page_reparse, title="再解析", icon=":material/refresh:", url_path="reparse"),
    "predict": st.Page(page_predict, title="予想", icon=":material/sports_score:", url_path="predict"),
    "train": st.Page(page_train, title="学習・評価", icon=":material/insights:", url_path="train"),
    "review": st.Page(page_review, title="回顧", icon=":material/history:", url_path="review"),
    "backtest": st.Page(page_backtest, title="シミュレーション", icon=":material/payments:", url_path="backtest"),
    "settings": st.Page(page_settings, title="設定", icon=":material/settings:", url_path="settings"),
}
st.navigation(list(PAGES.values()), position="top").run()
