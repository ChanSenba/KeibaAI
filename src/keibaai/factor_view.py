"""予想ファクターを、馬ごとの数字や文字で見られる形にする(画面の「ファクター」ページ用)。

学習はしない。出馬表の各馬について、過去の結果だけから作った値を並べる。
血統・ラップ・調教は取得したぶんだけ表示する。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .jvtraining import FAST_F4
from .predict import day_frame

# 画面に出す区分。cols は (列名, 画面での見出し)。
SECTIONS: list[dict] = [
    {"name": "脚質・馬齢・体重",
     "note": "脚質は過去3走の平均位置(頭数比)から推定: 逃げ ≦0.12 / 先行 ≦0.4 / 差し ≦0.7 / 追込。"
             "差し・追込で、レース中に位置を大きく上げる馬はマクリとしています。",
     "cols": [("style", "脚質(推定)"), ("early_pos_rel_3", "平均位置(頭数比)"), ("age", "馬齢"),
              ("body_weight", "馬体重"), ("body_weight_diff", "前走比"), ("bw_vs_avg", "平均体重との差"),
              ("weight_carried", "斤量"), ("weight_ratio", "斤量体重比"), ("weight_change", "斤量の増減")]},
    {"name": "適性(着度数)",
     "note": "今回と同じ条件での過去の成績を『1着-2着-3着-着外』で表します。距離は今回の±200mです。",
     "cols": [("rec_ct", "同じ芝/ダート"), ("rec_dist", "同じ距離帯"), ("rec_dir", "同じ回り"),
              ("rec_venue", "同じ競馬場"), ("rec_ground", "同じ馬場状態"), ("same_dist_top3_rate", "距離帯の3着内率")]},
    {"name": "近走・ローテーション",
     "note": "近5走は新しい順の着順です。叩き2走目などは、休み明け(90日以上)からの数え方です。",
     "cols": [("last5", "近5走の着順"), ("avg_finish_5", "5走の平均着順"), ("time_diff_5", "タイム差(5走平均)"),
              ("l3f_rank_3", "上がり順位(3走平均・頭数比)"), ("rotation", "ローテーション"),
              ("days_since_last", "前走からの日数")]},
    {"name": "騎手・厩舎",
     "note": "リーディングは、その年の1月から前日までの勝利数の順位です。成績は過去の結果だけで、当日分は含みません。",
     "cols": [("jockey", "騎手"), ("jockey_rank", "リーディング順位"), ("jockey_wins_year", "今年の勝利数"),
              ("jockey_ct_top3_rate", "コース種別の3着内率"), ("jockey_dist_top3_rate", "距離帯の3着内率"),
              ("jockey_venue_top3_rate", "競馬場の3着内率"), ("pair_n", "この馬との騎乗回数"),
              ("pair_top3_rate", "この馬との3着内率"), ("jockey_change", "乗り替わり"),
              ("tj_n", "厩舎×騎手の回数"), ("tj_top3_rate", "厩舎×騎手の3着内率")]},
    {"name": "馬場・枠・展開",
     "note": "トラックバイアスは直近10日以内の同じ競馬場・芝ダートの偏り(正なら先行有利・内有利)。"
             "ペース予想は、メンバーの脚質構成からの目安です。",
     "cols": [("bracket", "枠"), ("number", "馬番"), ("inner_pos", "内外(馬番/頭数)"),
              ("track_front_bias", "先行有利度"), ("track_inner_bias", "内有利度"), ("pace_forecast", "ペース予想"),
              ("n_front_runners", "先行寄りの馬の数"), ("class_change", "クラス変化"), ("distance_change", "距離の変化(m)"),
              ("dist_note", "距離短縮・延長")]},
    {"name": "オッズのゆがみ",
     "note": "騎手の単勝回収率が高いほど、その騎手は人気以上に走らせていて妙味があります。"
             "前走の人気 − 着順が大きいほど、前走は人気以上に走っています。",
     "cols": [("win_odds", "単勝オッズ"), ("popularity", "人気"), ("jockey_roi", "騎手の単勝回収率(倍)"),
              ("prev_pop_gap", "前走の人気 − 着順")]},
    {"name": "血統",
     "note": "父・母の父の産駒が、同じ条件で過去にどれだけ3着以内に来たか。血統を取得した馬だけ出ます(keibaai pedigree)。",
     "cols": [("sire", "父"), ("damsire", "母の父"), ("sire_line", "父の系統"), ("sire_top3_rate", "父産駒の3着内率"),
              ("sire_ctdist_top3_rate", "父産駒・芝ダ×距離帯"), ("sire_ground_top3_rate", "父産駒・芝ダ×馬場"),
              ("damsire_top3_rate", "母父産駒の3着内率"), ("damsire_ctdist_top3_rate", "母父産駒・芝ダ×距離帯")]},
    {"name": "調教",
     "note": "場所とタイムは JRA-VAN の坂路・ウッドの調教タイムから、レース前日までの直近14日の追い切り"
             "(坂路4F58秒以内・ウッド56秒以内)で作ります(取り込みは keibaai jv)。JRA-VAN にあるのは坂路とウッドだけです。"
             "短評・評価は netkeiba の最終追い切りで、取得したレースだけ出ます(keibaai training --date)。"
             "評価のメンバー内の差は、A=4〜D=1 とした点数からレースの平均を引いた値です。",
     "cols": [("train_center", "トレセン"), ("train_last_place", "最後の追い切り"),
              ("jv_last_f4", "最後の4F"), ("jv_last_l1", "最後の1F"),
              ("jv_last_f4_z", "コース基準との差(4F)"), ("jv_last_pct", "コース内の順位(4F)"),
              ("jv_last_accel", "加速(最初−最後の1F)"), ("jv_best_f4", "一番時計の4F"), ("train_best_place", "一番時計の場所"),
              ("jv_f4_vs_base", "ふだんとの差(4F)"), ("jv_f4_vs_field", "メンバー内の差(4F)"), ("jv_n14", "追い切りの本数(14日)"),
              ("jv_days_since", "最後の追い切りから(日)"), ("train_critic", "短評"), ("train_rank", "評価"),
              ("train_rank_vs_field", "評価のメンバー内の差")]},
    {"name": "ラップ・ペース",
     "note": "ペース差 = 前半3F − 後半3F(プラスは後傾=スロー寄り)。RPCI そのものではなく、結果ページのラップからの代用です。",
     "cols": [("prev_pace_diff", "前走のペース差"), ("pace_diff_3", "ペース差(3走平均)"),
              ("l3f_vs_lap_3", "自分の上がり−場の後半3F(3走平均)")]},
]

# 列ごとの説明と、良い向き。good は high(高いほど良い)/ low(低いほど良い)/ near0(0に近いほど良い)/
# record(着度数。3着内の回数が多いほど良い)/ last5(近走の着順の平均が小さいほど良い)/ None(良し悪しはない)
COLUMN_INFO: dict[str, tuple[str, str | None]] = {
    "style": ("過去3走の通過順から推定した脚質。展開との相性で見る", None),
    "early_pos_rel_3": ("過去3走の最初のコーナーの位置 ÷ 頭数の平均。0に近いほど前、1に近いほど後ろ", None),
    "age": ("馬の年齢", None),
    "body_weight": ("今回の馬体重(kg)", None),
    "body_weight_diff": ("前走からの馬体重の増減(kg)。大きく増減した馬は調子の変化に注意", "near0"),
    "bw_vs_avg": ("過去の平均体重との差(kg)。大きく離れているほど普段と違う状態", "near0"),
    "weight_carried": ("騎手が背負う重さ(kg)。軽いほど有利", "low"),
    "weight_ratio": ("斤量 ÷ 馬体重。小さいほど体に対して負担が軽い", "low"),
    "weight_change": ("前走からの斤量の増減(kg)。減っているほど楽になる", "low"),
    "rec_ct": ("今回と同じ芝/ダートでの『1着-2着-3着-着外』", "record"),
    "rec_dist": ("今回の距離±200mでの『1着-2着-3着-着外』", "record"),
    "rec_dir": ("今回と同じ回り(右/左)での『1着-2着-3着-着外』", "record"),
    "rec_venue": ("今回と同じ競馬場での『1着-2着-3着-着外』", "record"),
    "rec_ground": ("今回と同じ馬場状態(良・稍重など)での『1着-2着-3着-着外』", "record"),
    "same_dist_top3_rate": ("今回の距離帯で3着以内に来た割合。高いほど良い", "high"),
    "last5": ("近5走の着順(左が新しい)。小さい数字が並ぶほど好調", "last5"),
    "avg_finish_5": ("近5走の着順の平均。小さいほど良い", "low"),
    "time_diff_5": ("近5走の、勝ち馬とのタイム差(秒)の平均。小さいほど良い", "low"),
    "l3f_rank_3": ("過去3走の上がり3Fの順位 ÷ 頭数の平均。小さいほど末脚が速い", "low"),
    "rotation": ("前走からの間隔と、休み明けから何戦目か", None),
    "days_since_last": ("前走からの日数", None),
    "jockey": ("今回の騎手", None),
    "jockey_rank": ("今年の勝利数の順位。小さいほど上位の騎手", "low"),
    "jockey_wins_year": ("今年1月から前日までの勝利数。多いほど良い", "high"),
    "jockey_ct_top3_rate": ("騎手の、今回と同じ芝/ダートでの3着内率。高いほど良い", "high"),
    "jockey_dist_top3_rate": ("騎手の、今回の距離帯での3着内率。高いほど良い", "high"),
    "jockey_venue_top3_rate": ("騎手の、今回の競馬場での3着内率。高いほど良い", "high"),
    "pair_n": ("この馬に、この騎手が乗った回数", None),
    "pair_top3_rate": ("この馬とこの騎手の組み合わせでの3着内率。高いほど良い", "high"),
    "jockey_change": ("前走と騎手が変わったか", None),
    "tj_n": ("この厩舎の馬に、この騎手が乗った回数", None),
    "tj_top3_rate": ("この厩舎×この騎手の組み合わせでの3着内率。高いほど良い", "high"),
    "bracket": ("枠番。内外の有利不利は、コースと内有利度で変わる", None),
    "number": ("馬番", None),
    "inner_pos": ("馬番 ÷ 頭数。0に近いほど内、1に近いほど外", None),
    "track_front_bias": ("直近10日の同じ競馬場・芝ダの偏り。プラスなら前に付ける馬が有利(レース共通の値)", None),
    "track_inner_bias": ("直近10日の同じ競馬場・芝ダの偏り。プラスなら内の馬が有利(レース共通の値)", None),
    "pace_forecast": ("メンバーの脚質構成からのペースの目安(レース共通)", None),
    "n_front_runners": ("逃げ・先行寄りの馬の数(レース共通)。多いとペースが速くなりやすい", None),
    "class_change": ("前走からのクラスの変化。マイナスは格下げ(降級)で有利、プラスは昇級", "low"),
    "distance_change": ("前走からの距離の変化(m)。プラスは延長、マイナスは短縮", None),
    "dist_note": ("前走から300m以上の距離短縮・延長", None),
    "win_odds": ("単勝オッズ(倍)。低いほど人気=勝つと見られている", "low"),
    "popularity": ("単勝の人気順。小さいほど人気", "low"),
    "jockey_roi": ("騎手の単勝を買い続けたときの回収率(倍)。1を超えるほど人気以上に走らせていて妙味がある", "high"),
    "prev_pop_gap": ("前走の人気 − 着順。プラスが大きいほど人気以上に走った", "high"),
    "sire": ("父", None),
    "damsire": ("母の父", None),
    "sire_line": ("父の系統", None),
    "sire_top3_rate": ("父の産駒が3着以内に来た割合。高いほど良い", "high"),
    "sire_ctdist_top3_rate": ("父の産駒の、今回と同じ芝ダ×距離帯での3着内率。高いほど良い", "high"),
    "sire_ground_top3_rate": ("父の産駒の、今回と同じ芝ダ×馬場状態での3着内率。高いほど良い", "high"),
    "damsire_top3_rate": ("母の父の産駒が3着以内に来た割合。高いほど良い", "high"),
    "damsire_ctdist_top3_rate": ("母の父の産駒の、今回と同じ芝ダ×距離帯での3着内率。高いほど良い", "high"),
    "prev_pace_diff": ("前走の前半3F − 後半3F(秒)。プラスはスロー寄り、マイナスはハイペース寄り", None),
    "pace_diff_3": ("過去3走のペース差の平均。どんなペースを走ってきたか", None),
    "train_center": ("所属のトレセン(美浦・栗東)。直近14日の追い切りをした場所(JRA-VAN)", None),
    "train_last_place": ("最後の追い切りの場所。坂路か、ウッドのどのコースか(回り)。JRA-VAN になければ netkeiba のコース", None),
    "train_best_place": ("直近14日で一番時計を出した追い切りの場所", None),
    "train_critic": ("最終追い切りの短評(netkeiba)。気配上昇・仕上良好・前走並み など", None),
    "train_rank": ("最終追い切りの評価。A が最も良く、D に向かって下がる", None),
    "train_rank_vs_field": ("評価を A=4〜D=1 の点にして、レース平均を引いた値。プラスほどメンバーの中で良い", "high"),
    "jv_n14": ("レース前14日の追い切り(速い調教)の本数。多いほど仕上げを重ねている", None),
    "jv_best_f4": ("直近14日の一番時計の4F(秒)。小さいほど速い", "low"),
    "jv_best_l1": ("直近14日で一番速かった最後の1F(秒)。小さいほど終いが速い", "low"),
    "jv_last_f4": ("直近14日の最後の追い切りの4F(秒)。小さいほど速い", "low"),
    "jv_last_l1": ("直近14日の最後の追い切りの最後の1F(秒)。小さいほど速い", "low"),
    "jv_days_since": ("最後の追い切りから、レースまでの日数", None),
    "jv_f4_vs_base": ("直近14日の一番時計の4F − 15〜180日前の追い切りの平均(秒)。マイナスは、ふだんより速い", "low"),
    "jv_f4_vs_field": ("一番時計の4F − レース平均(秒)。マイナスは、メンバーの中で速い", "low"),
    "jv_best_f4_z": ("直近14日の一番時計の4F − 同じトレセン・コース・時期の追い切りの中央値(秒)。マイナスは、そのコースの標準より速い", "low"),
    "jv_best_pct": ("直近14日の一番時計の、同じトレセン・コース・時期の中での順位(0が最速、1が最遅)", "low"),
    "jv_last_f4_z": ("最後の追い切りの4F − 同じトレセン・コース・時期の中央値(秒)。マイナスは、標準より速い", "low"),
    "jv_last_pct": ("最後の追い切りの4Fの、同じトレセン・コース・時期の中での順位(0が最速、1が最遅)", "low"),
    "jv_last_f2_z": ("最後の追い切りの2F − 同じコース・時期の中央値(秒)。マイナスは、終いが速い", "low"),
    "jv_last_l1_z": ("最後の追い切りの最後の1F − 同じコース・時期の中央値(秒)。マイナスは、終いが速い", "low"),
    "jv_last_accel": ("最後の追い切りの 最初の1Fのラップ − 最後の1Fのラップ(秒)。プラスは加速、マイナスは失速", "high"),
    "l3f_vs_lap_3": ("自分の上がり3F − レース全体の後半3F(3走平均・秒)。マイナスが大きいほど周りより速く上がった", "low"),
}

GOOD_LABEL = {"high": "高いほど良い", "low": "低いほど良い", "near0": "0に近いほど良い",
              "record": "3着内の回数が多いほど良い", "last5": "小さい着順が多いほど良い"}


def _good_score(values: pd.Series, good: str) -> pd.Series:
    """大きいほど良い、に揃えた点数。比べられない値は NaN。"""
    if good == "high":
        return pd.to_numeric(values, errors="coerce")
    if good == "low":
        return -pd.to_numeric(values, errors="coerce")
    if good == "near0":
        return -pd.to_numeric(values, errors="coerce").abs()
    if good == "record":
        def rec(v):
            try:
                a, b, c, _ = (int(x) for x in str(v).split("-"))
            except ValueError:
                return np.nan
            top3 = a + b + c
            return top3 + a / 100 + b / 10000 if top3 else np.nan  # 3着内の回数、同じなら1着・2着の多い順
        return values.map(rec)
    if good == "last5":
        def avg(v):
            nums = [int(x) for x in str(v).split("-") if x.isdigit()]
            return -sum(nums) / len(nums) if nums else np.nan
        return values.map(avg)
    return pd.Series(np.nan, index=values.index)


def top_rank(values: pd.Series, good: str | None, n: int = 5) -> pd.Series:
    """良い値の順位(1が最良)。上位 n 頭(同じ値は同順位)だけ入り、ほかは NaN。

    頭数が少ないときは半分まで。全馬が同じ値なら、差がないので順位をつけない。
    """
    if good is None:
        return pd.Series(np.nan, index=values.index)
    score = _good_score(values, good)
    known = score.dropna()
    if known.nunique() < 2:
        return pd.Series(np.nan, index=values.index)
    n = min(n, -(-len(known) // 2))
    rank = score.rank(method="min", ascending=False)
    return rank.where(rank <= n)


# 「総合」タブ用。各区分から、予想に効きやすい数値を抜き出したもの(見出しは区分名つき)
SUMMARY: dict = {
    "note": "各区分から重要な数値を1〜2個ずつ抜き出した表です。色は、その列で良い値の上位の馬。"
            "血統・ラップは取得した馬だけ値が入ります。くわしい数値は、各区分のタブで見られます。",
    "cols": [("style", "脚質"), ("avg_finish_5", "5走の平均着順"), ("time_diff_5", "タイム差(5走平均)"),
             ("l3f_rank_3", "上がり順位(3走)"), ("rec_dist", "同じ距離帯"), ("rec_ground", "同じ馬場状態"),
             ("rotation", "ローテーション"), ("jockey_rank", "騎手リーディング"),
             ("jockey_ct_top3_rate", "騎手の3着内率"), ("pair_top3_rate", "この馬との3着内率"),
             ("class_change", "クラス変化"), ("prev_pop_gap", "前走の人気 − 着順"),
             ("sire_ctdist_top3_rate", "父産駒・芝ダ×距離帯"), ("l3f_vs_lap_3", "上がり−場の後半3F"),
             ("popularity", "人気"), ("win_odds", "単勝オッズ")],
}

STYLES = ["逃げ", "先行", "差し", "追込"]


def style_label(early_pos_rel_3: float, pos_gain_3: float) -> str:
    """過去3走の平均位置(頭数比)から脚質を推定する。履歴がなければ「不明」。"""
    if early_pos_rel_3 != early_pos_rel_3:
        return "不明"
    if early_pos_rel_3 <= 0.12:
        return "逃げ"
    if early_pos_rel_3 <= 0.4:
        return "先行"
    label = "差し" if early_pos_rel_3 <= 0.7 else "追込"
    if pos_gain_3 == pos_gain_3 and pos_gain_3 >= 0.25:
        return "マクリ"
    return label


def rotation_label(days: float, runs_since_layoff: float) -> str:
    if days != days:
        return "初出走"
    if days <= 8:
        return "連闘"
    if days >= 90:
        return f"休み明け({days / 7:.0f}週ぶり)"
    n = int(runs_since_layoff)
    if n == 2:
        return "叩き2走目"
    if n == 3:
        return "叩き3走目"
    return f"中{days / 7:.0f}週"


def pace_forecast(styles: pd.Series) -> str:
    """逃げ・先行の数からの目安。逃げ馬がいない/多いでペースが変わる。"""
    known = styles[styles != "不明"]
    if known.empty:
        return "不明"
    n_nige = int((known == "逃げ").sum())
    n_front = n_nige + int((known == "先行").sum())
    if n_nige >= 2 or n_front / len(known) >= 0.45:
        return "ハイペース寄り"
    if n_nige == 0 and n_front / len(known) < 0.25:
        return "スロー寄り"
    return "平均的"


def _record(finishes: pd.Series) -> str:
    f = finishes.dropna()
    return f"{int((f == 1).sum())}-{int((f == 2).sum())}-{int((f == 3).sum())}-{int((f > 3).sum())}"


def _records(target: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """各馬の、今回と同じ条件での過去の着度数。"""
    by_horse = {h: g for h, g in hist.groupby("horse_id", sort=False)}
    rows = []
    for _, t in target.iterrows():
        g = by_horse.get(t["horse_id"])
        if g is None:
            rows.append({k: "0-0-0-0" for k in ("rec_ct", "rec_dist", "rec_dir", "rec_venue", "rec_ground")})
            continue
        rows.append({
            "rec_ct": _record(g.loc[g["course_type"] == t["course_type"], "finish"]),
            "rec_dist": _record(g.loc[(g["distance"] - t["distance"]).abs() <= 200, "finish"]),
            "rec_dir": _record(g.loc[g["direction"] == t["direction"], "finish"]) if isinstance(t["direction"], str)
            else "—",
            "rec_venue": _record(g.loc[g["venue"] == t["venue"], "finish"]),
            "rec_ground": _record(g.loc[g["ground"] == t["ground"], "finish"]),
        })
    return pd.DataFrame(rows, index=target.index)


def _last5(target: pd.DataFrame, hist: pd.DataFrame) -> pd.Series:
    h = hist.sort_values("date", kind="stable")
    recent = {k: "-".join(str(int(x)) for x in g["finish"].dropna().tail(5)[::-1])
              for k, g in h.groupby("horse_id", sort=False)}
    return target["horse_id"].map(recent).fillna("—")


def _jockey_year_wins(hist: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """その年の1月から前日までの騎手別の勝利数と、その順位(同数は同順位)。"""
    y = hist[(hist["date"].dt.year == date.year) & (hist["date"] < date)]
    wins = (y.assign(w=(y["finish"] == 1).astype(int)).groupby("jockey_id")["w"].sum())
    out = pd.DataFrame({"jockey_wins_year": wins})
    out["jockey_rank"] = out["jockey_wins_year"].rank(method="min", ascending=False)
    return out


def _pedigree_names(conn: sqlite3.Connection, horse_ids) -> pd.DataFrame:
    ids = [h for h in set(horse_ids) if isinstance(h, str)]
    if not ids:
        return pd.DataFrame(columns=["horse_id", "sire", "damsire", "sire_line"])
    q = ",".join("?" * len(ids))
    return pd.read_sql_query(
        f"SELECT horse_id, sire, damsire, sire_line FROM pedigree WHERE horse_id IN ({q})", conn, params=ids)


CENTERS = {"0": "美浦", "1": "栗東"}
WOOD_COURSES = "ABCDE"  # JRA-VAN のウッドのコース(0〜4 が A〜E)。栗東の C が CW


def training_place(kind: str, center: str | None, course: str | None, turn: str | None) -> str:
    """JRA-VAN の調教1本の場所。坂路 / CW(右) / ウッドD(左) など。"""
    if kind == "slope":
        return "坂路"
    letter = WOOD_COURSES[int(course)] if course and course.isdigit() and int(course) < len(WOOD_COURSES) else ""
    name = "CW" if center == "1" and letter == "C" else f"ウッド{letter}"
    return name + {"0": "(右)", "1": "(左)"}.get(turn or "", "")


def _training_places(conn: sqlite3.Connection, t: pd.DataFrame, date: str) -> pd.DataFrame:
    """出走馬ごとに、トレセン・最後の追い切りと一番時計の場所(JRA-VAN)。JRA-VAN にない馬は netkeiba のコース。
    調教の特徴量と同じく、レース前日までの直近14日の追い切りで見る。
    追い方・併せ馬は、netkeiba の未ログインのページでは各レースの先頭3頭しか出ず取っていないので、出さない。"""
    out = pd.DataFrame(index=t.index, columns=["train_center", "train_last_place", "train_best_place"], dtype=object)
    ids = [h for h in set(t["horse_id"]) if isinstance(h, str)]
    day = pd.Timestamp(date)
    if ids:
        q = ",".join("?" * len(ids))
        tr = pd.read_sql_query(
            f"SELECT horse_id, date, time, kind, center, course, turn, f4 FROM jv_training "
            f"WHERE horse_id IN ({q}) AND date >= ? AND date < ? AND f4 IS NOT NULL", conn,
            params=[*ids, (day - pd.Timedelta(days=14)).date().isoformat(), day.date().isoformat()])
        tr = tr[tr["f4"] <= tr["kind"].map(FAST_F4)].sort_values(["date", "time"])
        if not tr.empty:
            tr["place"] = [training_place(*r) for r in tr[["kind", "center", "course", "turn"]].itertuples(index=False)]
            last = tr.groupby("horse_id").last()
            best = tr.loc[tr.groupby("horse_id")["f4"].idxmin()].set_index("horse_id")
            out["train_center"] = t["horse_id"].map(last["center"].map(CENTERS))
            out["train_last_place"] = t["horse_id"].map(last["place"])
            out["train_best_place"] = t["horse_id"].map(best["place"])
    races = list(set(t["race_id"]))
    nk = pd.read_sql_query(
        f"SELECT race_id, number, course FROM training WHERE seq = 0 "
        f"AND race_id IN ({','.join('?' * len(races))})", conn, params=races) if races else pd.DataFrame()
    if not nk.empty:
        got = t[["race_id", "number"]].merge(nk, on=["race_id", "number"], how="left")
        got.index = t.index
        out["train_last_place"] = out["train_last_place"].fillna(got["course"])  # JRA-VAN にない馬は netkeiba のコース
    return out


def factor_values(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    """date の出馬表の各馬について、SECTIONS の列をすべて持つ表を返す(race_id・馬名つき)。"""
    df, hist = day_frame(conn, date)
    t = df[df["is_entry"]].copy()
    ts = pd.Timestamp(date)

    t["style"] = [style_label(a, b) for a, b in zip(t["early_pos_rel_3"], t["pos_gain_3"])]
    t["pace_forecast"] = t.groupby("race_id")["style"].transform(pace_forecast)
    mine = hist[hist["horse_id"].isin(set(t["horse_id"]))]  # 出走馬の過去だけ。全馬の履歴を分けるのは無駄
    t = t.join(_records(t, mine)).assign(last5=_last5(t, mine))
    t["rotation"] = [rotation_label(d, r) for d, r in zip(t["days_since_last"], t["runs_since_layoff"])]
    t["jockey_change"] = t["jockey_changed"].map({1.0: "乗り替わり", 0.0: "継続"}).fillna("—")
    t["dist_note"] = np.select([t["dist_shorten300"] == 1, t["dist_extend300"] == 1], ["短縮300m以上", "延長300m以上"], "—")
    yj = _jockey_year_wins(hist, ts)
    t = t.merge(yj, left_on="jockey_id", right_index=True, how="left")
    t["jockey_wins_year"] = t["jockey_wins_year"].fillna(0)  # 今年まだ騎乗がない
    t = t.merge(_pedigree_names(conn, t["horse_id"]), on="horse_id", how="left")
    t["jockey"] = t["jockey"].astype(object)
    t["train_critic"] = t["train_critic"].astype(object)
    t = t.join(_training_places(conn, t, date))
    t["venue"] = t["venue"].astype(str)
    keep = ["race_id", "venue", "round", "horse_name", *dict.fromkeys(c for s in SECTIONS for c, _ in s["cols"])]
    return t[[c for c in keep if c in t]].sort_values(["race_id", "number"]).reset_index(drop=True)


def section_status(conn: sqlite3.Connection) -> list[dict]:
    """各区分のデータが揃っているか。state は ready / partial / missing。"""
    n_horses = conn.execute("SELECT COUNT(DISTINCT horse_id) FROM results").fetchone()[0]
    n_ped = conn.execute("SELECT COUNT(*) FROM pedigree").fetchone()[0]
    n_races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    n_flat = conn.execute("SELECT COUNT(*) FROM races WHERE course_type IN ('芝','ダ')").fetchone()[0]
    n_lap = conn.execute("SELECT COUNT(*) FROM races WHERE course_type IN ('芝','ダ') AND pace_first3f IS NOT NULL").fetchone()[0]
    out = []
    for s in SECTIONS:
        name = s["name"]
        if name == "調教":
            n_jv = conn.execute("SELECT COUNT(*) FROM jv_training").fetchone()[0]
            n_tr = conn.execute("SELECT COUNT(*) FROM training_fetched WHERE race_id IN (SELECT race_id FROM races)"
                                ).fetchone()[0]
            nk = f"netkeiba の評価 {n_tr:,} / {n_races:,} レース"
            state = "ready" if n_jv else "partial" if n_tr else "missing"
            text = (f"JRA-VAN の調教タイム {n_jv:,} 本・{nk}" if n_jv else
                    nk + "(場所とタイムは keibaai jv で取得)" if n_tr else "未取得(keibaai jv で取得)")
        elif name == "血統":
            share = n_ped / n_horses if n_horses else 0
            state = "ready" if share >= 0.95 else "partial" if n_ped else "missing"
            text = f"{n_ped:,} / {n_horses:,} 頭ぶん取得済み" if n_ped else "未取得(keibaai jv --what pedigree で取得)"
        elif name == "ラップ・ペース":
            share = n_lap / n_flat if n_flat else 0
            state = "ready" if share >= 0.95 else "partial" if n_lap else "missing"
            text = f"{n_lap:,} / {n_flat:,} レース" if n_lap else "未取得(keibaai reparse で読み込み)"
        else:
            state, text = "ready", "取得済み"
        out.append({"name": name, "state": state, "text": text})
    return out


GATES = ["内", "中", "外"]


def gate_bias(conn: sqlite3.Connection, race, before: str) -> pd.DataFrame:
    """このコースと、この A〜E コースでの、内・中・外の枠の過去の 3 着内率(参考。予想には使っていない)。

    枠は馬番を頭数で3等分して、内(前の3分の1)・中・外とする。before(YYYY-MM-DD)より前のレースだけを数える。
    行は「同じコース」(競馬場・芝ダ・距離・内外回り)と「同じ A〜E」(競馬場・芝ダ・柵)。分からない行は出さない。
    """
    venue, ct = race.get("venue"), race.get("course_type")
    if not isinstance(venue, str) or not isinstance(ct, str):
        return pd.DataFrame()
    df = pd.read_sql_query(
        "SELECT r.race_id, r.number, r.finish, c.distance, c.course, c.rail FROM results r JOIN races c USING (race_id) "
        "WHERE c.venue = ? AND c.course_type = ? AND c.date < ? "
        "AND (r.finish IS NOT NULL OR r.status IN ('中', '失'))",
        conn, params=(venue, ct, before))
    if df.empty:
        return pd.DataFrame()
    pos = df["number"] / df.groupby("race_id")["number"].transform("count")
    df["gate"] = np.select([pos <= 1 / 3, pos <= 2 / 3], GATES[:2], GATES[2])
    df["top3"] = (df["finish"] <= 3).astype(float)
    same = lambda col, v: df[col].isna() if v is None or v != v else df[col] == v
    rows = []
    distance = pd.to_numeric(race.get("distance"), errors="coerce")
    if distance == distance:
        rows.append(("同じコース", df[(df["distance"] == distance) & same("course", race.get("course"))]))
    rail = race.get("rail")
    if isinstance(rail, str) and rail:
        rows.append((f"同じ {rail}コース", df[df["rail"] == rail]))
    out = []
    for name, g in rows:
        if g.empty:
            continue
        rate = g.groupby("gate")["top3"].mean()
        row = {"条件": name, "レース数": g["race_id"].nunique()}
        row.update({f"{k}枠": rate.get(k, np.nan) for k in GATES})
        row["内−外"] = (row["内枠"] - row["外枠"]) * 100
        out.append(row)
    return pd.DataFrame(out)
