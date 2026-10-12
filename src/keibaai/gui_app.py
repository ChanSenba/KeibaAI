"""KeibaAI の操作画面(Streamlit)。`keibaai-gui` で起動する。"""
from __future__ import annotations

import contextlib
import datetime as dt
from pathlib import Path

import streamlit as st

from keibaai import db, gui
from keibaai.cli import reparse_cache
from keibaai.fetch import Fetcher

st.set_page_config(page_title="KeibaAI", page_icon="🏇", layout="wide")


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
        self.box.code("\n".join(self.lines[-self.max_lines:]) or " ", language=None)
        return len(s)

    def flush(self) -> None:
        pass


with st.sidebar:
    st.title("🏇 KeibaAI")
    page = st.radio("メニュー", ["データ確認", "データ取得", "再解析", "学習・評価"], label_visibility="collapsed")
    st.divider()
    db_path = st.text_input("DB ファイル", "data/keiba.db")
    cache_dir = st.text_input("キャッシュフォルダ", "data/cache")
    st.caption(f"作業フォルダ: {Path.cwd()}")

conn = db.connect(db_path)


if page == "データ確認":
    st.header("取得済みデータ")
    s = gui.summary(conn)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("レース数", f"{s['races']:,}")
    c2.metric("出走データ数", f"{s['results']:,}")
    c3.metric("期間", f"{s['first']} 〜 {s['last']}" if s["first"] else "なし")
    if s["races"] == 0:
        st.info("まだデータがありません。左の「データ取得」から期間を指定して取得してください。")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("月別レース数")
            st.bar_chart(gui.races_by_month(conn), x="month", y="races", x_label="", y_label="レース数")
        with right:
            st.subheader("競馬場別")
            st.dataframe(gui.races_by_venue(conn), hide_index=True, width="stretch",
                         column_config={"venue": "競馬場", "races": "レース数"})
        st.subheader("最近のレース")
        st.dataframe(gui.recent_races(conn), hide_index=True, width="stretch", column_config={
            "date": "日付", "venue": "競馬場", "round": "R", "name": "レース名", "course_type": "芝/ダ",
            "distance": "距離", "ground": "馬場", "runners": "頭数"})


elif page == "データ取得":
    st.header("データ取得")
    st.write("期間内の土日・月曜の JRA レース結果を netkeiba から取得して DB に保存します。"
             "取得済みのレースは飛ばすので、同じ期間を何度実行しても大丈夫です。")
    today = dt.date.today()
    c1, c2 = st.columns(2)
    start = c1.date_input("開始日", today - dt.timedelta(days=14), format="YYYY-MM-DD")
    end = c2.date_input("終了日", today, format="YYYY-MM-DD")
    days = gui.race_days(start, end) if start <= end else []
    st.caption(f"対象日 {len(days)} 日。1ページごとに3秒以上あけるため、1日(約36レース)でおよそ2分かかります。")
    st.warning("取得中にほかのボタンやメニューを押すと取得が止まります(それまでに取得した分は保存済みです)。")
    if st.button("取得開始", type="primary", disabled=not days):
        bar = st.progress(0.0, "準備中")
        log = LogBox()

        def on_day(i, n, d):
            bar.progress(i / n if n else 1.0, f"{d} を取得中 ({i + 1}/{n} 日目)" if d else "完了")

        try:
            with contextlib.redirect_stdout(log):
                total = gui.collect(conn, Fetcher(cache_dir=cache_dir), start, end, on_day)
            st.success(f"完了: {total} レースを新しく保存しました。")
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:  # 通信エラーなど。途中までの分は保存されている
            st.error(f"取得が止まりました: {e}")


elif page == "再解析":
    st.header("再解析(通信なし)")
    st.write("保存済みのページ(キャッシュ)を解析し直して DB を更新します。"
             "プログラムを更新したあとに実行してください。")
    n_cached = len(list((Path(cache_dir) / "race").glob("*.html")))
    st.caption(f"キャッシュ済みページ: {n_cached:,} 件")
    if st.button("再解析を実行", type="primary", disabled=n_cached == 0):
        log = LogBox()
        with st.spinner("解析中"), contextlib.redirect_stdout(log):
            ok, bad = reparse_cache(conn, cache_dir)
        st.success(f"完了: {ok} レースを更新、{bad} 件は飛ばしました。")


elif page == "学習・評価":
    st.header("学習・評価")
    first, last = gui.date_bounds(conn)
    if first is None:
        st.info("まだデータがありません。先に「データ取得」をしてください。")
        st.stop()
    st.write(f"取得済み期間 {first} 〜 {last} を、評価開始日より前(学習用)と以後(評価用)に分けて学習します。")
    default = max(first, last - dt.timedelta(days=90))
    test_from = st.date_input("評価開始日", default, min_value=first, max_value=last, format="YYYY-MM-DD")
    if st.button("学習して評価", type="primary"):
        try:
            with st.spinner("学習中(データ量によって数分かかります)"):
                st.session_state["report"] = (test_from, gui.train_reports(conn, test_from.isoformat()))
        except ValueError as e:
            st.error(str(e))
    if "report" in st.session_state:
        when, report = st.session_state["report"]
        st.subheader(f"評価結果(評価開始日 {when})")
        st.dataframe(report, hide_index=True, width="stretch", column_config={
            "勝LogLoss": st.column_config.NumberColumn(format="%.4f"),
            "勝AUC": st.column_config.NumberColumn(format="%.3f"),
            "複AUC": st.column_config.NumberColumn(format="%.3f"),
            "1番手的中": st.column_config.NumberColumn(format="percent"),
            "単勝回収率(%)": st.column_config.NumberColumn(format="%.1f"),
        })
        st.markdown(
            "- **基準**: 毎回1番人気を単勝で買った場合\n"
            "- **オッズなし**: 馬・騎手・調教師の過去成績とレース条件だけで予測\n"
            "- **オッズあり**: 上に単勝オッズ・人気を加えたもの\n\n"
            "勝AUC は 0.5 が当てずっぽうで、高いほどよい。単勝の控除率は約20%なので、回収率が100%を超えるのは簡単ではありません。"
        )
