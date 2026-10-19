"""予想ファクターを、馬ごとの数字や文字で見られる形にする(画面の「ファクター」ページ用)。

学習はしない。出馬表の各馬について、過去の結果だけから作った値を並べる。
血統・ラップ・調教は取得したぶんだけ表示する。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .features import inner_outer, same_course
from .jvtraining import FAST_F4
from .predict import _cached, day_frame

# 画面に出す区分。cols は (列名, 画面での見出し)。
SECTIONS: list[dict] = [
    {"name": "脚質・馬齢・体重",
     "note": "脚質は過去3走の平均位置(頭数比)から推定: 逃げ ≦0.12 / 先行 ≦0.4 / 差し ≦0.7 / 追込。"
             "差し・追込で、レース中に位置を大きく上げる馬はマクリとしています。",
     "cols": [("style", "脚質(推定)"), ("early_pos_rel_3", "平均位置(頭数比)"), ("age", "馬齢"),
              ("body_weight", "馬体重"), ("body_weight_diff", "前走比"), ("bw_vs_avg", "平均体重との差"),
              ("weight_carried", "斤量"), ("weight_ratio", "斤量体重比"), ("weight_change", "斤量の増減")]},
    {"name": "適性(着度数)",
     "note": "今回と同じ条件での過去の成績を『1着-2着-3着-着外』で表します。"
             "同コースは、同じ競馬場で芝ダ・内回り外回り・距離が同じコースです。同じ距離は、競馬場を問わず距離がちょうど同じ走り、"
             "距離帯は今回の±200mです。",
     "cols": [("rec_course", "同コース"), ("rec_same_dist", "同じ距離"), ("rec_ct", "同じ芝/ダート"), ("rec_dist", "同じ距離帯"), ("rec_dir", "同じ回り"),
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
             "ペース予想は、メンバーの脚質構成からの目安です。"
             "同コースの直近は、同じ競馬場で芝ダ・内回り外回り・距離が同じコースの、直近5開催日(この日より前)の平均です。"
             "展開適性は、この馬の脚質が、ペース予想と同コース直近の展開に合っているかの点(プラスは合う)です。",
     "cols": [("bracket", "枠"), ("number", "馬番"), ("inner_pos", "内外(馬番/頭数)"),
              ("track_front_bias", "先行有利度"), ("track_inner_bias", "内有利度"), ("pace_forecast", "ペース予想"),
              ("course_pace_diff_5", "同コース直近のペース差"), ("course_front_edge_5", "同コース直近の先行有利度"),
              ("pace_fit", "展開適性"),
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
             "騎乗者・脚色・併せ馬は netkeiba の調教の詳細からで、未ログインでは各レースの先頭3頭しか取れないため、取れた馬だけ出ます。"
             "評価のメンバー内の差は、A=4〜D=1 とした点数からレースの平均を引いた値です。",
     "cols": [("train_center", "トレセン"), ("train_last_place", "最後の追い切り"),
              ("jv_last_f4", "最後の4F"), ("jv_last_l1", "最後の1F"),
              ("jv_last_f4_z", "コース基準との差(4F)"), ("jv_last_pct", "コース内の順位(4F)"),
              ("jv_last_accel", "加速(最初−最後の1F)"),
              ("jv_last_lap1", "ラップ1(4F→3F)"), ("jv_last_lap2", "ラップ2(3F→2F)"), ("jv_last_lap3", "ラップ3(2F→1F)"),
              ("jv_last_lap1_z", "ラップ1の基準差"), ("jv_last_lap2_z", "ラップ2の基準差"), ("jv_last_lap3_z", "ラップ3の基準差"),
              ("jv_last_l1_z", "終い1Fの基準差"), ("jv_best_f4", "一番時計の4F"), ("train_best_place", "一番時計の場所"),
              ("jv_f4_vs_base", "ふだんとの差(4F)"), ("jv_f4_vs_field", "メンバー内の差(4F)"), ("jv_n14", "追い切りの本数(14日)"),
              ("jv_days_since", "最後の追い切りから(日)"), ("train_day", "最終追い切りの日"), ("train_rider", "騎乗者"),
              ("train_load", "脚色"), ("train_partner", "併せ馬"), ("train_critic", "短評"), ("train_rank", "評価"),
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
    "rec_course": ("同コース(同じ競馬場で、芝ダ・内回り外回り・距離が今回と同じ)での『1着-2着-3着-着外』", "record"),
    "rec_same_dist": ("今回と同じ距離(競馬場は問わない)での『1着-2着-3着-着外』。同じ距離での経験", "record"),
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
    "train_day": ("netkeiba の調教の詳細から取れた、最終追い切りの日。詳細は会員限定で、取れた馬だけ出ます", None),
    "train_rider": ("最終追い切りに乗った人(助手・騎手など)。netkeiba の調教の詳細から取れた馬だけ出ます", None),
    "train_load": ("最終追い切りの脚色(馬也・強め・一杯など)。netkeiba の調教の詳細から取れた馬だけ出ます", None),
    "train_partner": ("最終追い切りの併せ馬(相手・脚色・先着か遅れか)。netkeiba の調教の詳細から取れた馬だけ出ます。空欄は併せ馬なしか未取得", None),
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
    "course_pace_diff_5": ("同コース(同じ競馬場の、芝ダ・内回り外回り・距離が同じコース)の直近5開催日の、前半3F − 後半3Fの平均(秒)。プラスは後傾=スロー、マイナスはハイペースが続いている", None),
    "course_front_edge_5": ("同コースの直近5開催日の、前に付けた馬の3着内率 − ほかの馬の率の平均。プラスは先行有利、マイナスは差し有利", None),
    "pace_fit": ("この馬の脚質が、ペース予想と同コース直近の展開に合うか。プラスは合う(スローや先行有利なら逃げ・先行、ハイペースや差し有利なら差し・追込)", "high"),
    "jv_last_lap1": ("最後の追い切りの最初の1F(4F−3F)のラップ(秒)。長い追い切りの出だしの速さ", None),
    "jv_last_lap2": ("最後の追い切りの2つ目の1F(3F−2F)のラップ(秒)", None),
    "jv_last_lap3": ("最後の追い切りの3つ目の1F(2F−最後の1F)のラップ(秒)", None),
    "jv_last_lap1_z": ("最初の1Fのラップ − 同じトレセン・コース・時期の中央値(秒)。マイナスは、出だしから速い", None),
    "jv_last_lap2_z": ("2つ目の1Fのラップ − 同じコース・時期の中央値(秒)。マイナスは、中間も速い", None),
    "jv_last_lap3_z": ("3つ目の1Fのラップ − 同じコース・時期の中央値(秒)。マイナスは、4コーナーからの加速が速い", "low"),
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
            "血統・ラップは取得した馬だけ値が入ります。くわしい数値は、各区分のタブで見られます。"
            "左の「総合点」は、人気・オッズを除く列を1つずつ点数にして足したものです(下の「点数のつけ方」)。",
    "cols": [("style", "脚質"), ("avg_finish_5", "5走の平均着順"), ("time_diff_5", "タイム差(5走平均)"),
             ("l3f_rank_3", "上がり順位(3走)"), ("rec_dist", "同じ距離帯"), ("rec_ground", "同じ馬場状態"),
             ("rotation", "ローテーション"), ("jockey_rank", "騎手リーディング"),
             ("jockey_ct_top3_rate", "騎手の3着内率"), ("pair_top3_rate", "この馬との3着内率"),
             ("class_change", "クラス変化"), ("prev_pop_gap", "前走の人気 − 着順"),
             ("sire_ctdist_top3_rate", "父産駒・芝ダ×距離帯"), ("l3f_vs_lap_3", "上がり−場の後半3F"),
             ("jv_last_f4_z", "追い切り4F(基準差)"), ("jv_last_lap3_z", "追い切りラップ3(基準差)"),
             ("jv_last_l1_z", "追い切り終い1F(基準差)"), ("pace_fit", "展開適性"),
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


# 脚質ごとの、前に付ける度合い(逃げ +1 … 追込 −1)。展開適性の計算に使う
STYLE_FRONT = {"逃げ": 1.0, "先行": 0.5, "差し": -0.5, "追込": -1.0, "マクリ": -0.5, "不明": 0.0}
PACE_SLOW, PACE_HIGH = 1.0, -1.0  # ペース差(前半3F−後半3F)がこれ以上ならスロー、以下ならハイペースとする


def pace_label(diff: float) -> str:
    """ペース差(前半3F − 後半3F。プラスは後傾)の呼び名。"""
    if diff != diff:
        return "不明"
    return "スロー" if diff >= PACE_SLOW else "ハイペース" if diff <= PACE_HIGH else "平均的"


def front_edge(forecast: str, course_pace_diff: float, course_front_edge: float) -> float:
    """そのレースで前に付ける馬が有利か(−1〜+1)。メンバーの脚質構成からのペース予想(半分)と、
    同コースの直近の展開(ペース差・先行有利度、各4分の1)を合わせる。分からない部分は0とする。"""
    field = {"スロー寄り": 1.0, "ハイペース寄り": -1.0}.get(forecast, 0.0)
    pace = 0.0 if course_pace_diff != course_pace_diff else float(np.clip(course_pace_diff / 1.5, -1, 1))
    edge = 0.0 if course_front_edge != course_front_edge else float(np.clip(course_front_edge / 0.1, -1, 1))
    return 0.5 * field + 0.25 * pace + 0.25 * edge


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
            rows.append({k: "0-0-0-0" for k in ("rec_course", "rec_same_dist", "rec_ct", "rec_dist", "rec_dir", "rec_venue",
                                                 "rec_ground")})
            continue
        rows.append({
            "rec_course": _record(g.loc[[same_course(r, t) for _, r in g.iterrows()], "finish"]),
            "rec_same_dist": _record(g.loc[g["distance"] == t["distance"], "finish"]),
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
    騎乗者・脚色・併せ馬は、netkeiba の未ログインのページでは各レースの先頭3頭しか出ないので、取れた馬だけ入る。"""
    out = pd.DataFrame(index=t.index, dtype=object, columns=[
        "train_center", "train_last_place", "train_best_place", "train_day", "train_rider", "train_load", "train_partner"])
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
        f"SELECT race_id, number, course, date, rider, load, partner FROM training WHERE seq = 0 "
        f"AND race_id IN ({','.join('?' * len(races))})", conn, params=races) if races else pd.DataFrame()
    if not nk.empty:
        got = t[["race_id", "number"]].merge(nk, on=["race_id", "number"], how="left")
        got.index = t.index
        out["train_last_place"] = out["train_last_place"].fillna(got["course"])  # JRA-VAN にない馬は netkeiba のコース
        for col, src in (("train_day", "date"), ("train_rider", "rider"), ("train_load", "load"), ("train_partner", "partner")):
            out[col] = got[src].astype(object)  # 詳細が取れた馬だけ入る(取れていなければ空)
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
    edge = [front_edge(f, p, e) for f, p, e in zip(t["pace_forecast"], t["course_pace_diff_5"], t["course_front_edge_5"])]
    t["pace_fit"] = t["style"].map(STYLE_FRONT).to_numpy() * np.array(edge)
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


def factor_values_cached(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    """factor_values の結果を DB の隣に保存し、データが同じなら次からはそれを返す(予想と同じしくみ)。

    作り直しに過去の全成績の特徴量づくりがかかるので、画面を開き直したりオッズを取り直したりしても待たずに済ませる。
    血統・調教の取得で件数が変わったときも、作り直す。
    """
    counts = tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                   for t in ("pedigree", "jv_training", "training_fetched"))
    return _cached(conn, "factors", date, True, counts, lambda: factor_values(conn, date)).copy()


def section_status(conn: sqlite3.Connection) -> list[dict]:
    """各区分のデータが揃っているか。state は ready / partial / missing。"""
    n_horses = conn.execute("SELECT COUNT(DISTINCT horse_id) FROM results").fetchone()[0]
    # 血統の表には成績に出てこない馬(JRA-VAN の全登録馬)も入っているので、成績の馬と重なる分だけ数える
    n_ped = conn.execute("SELECT COUNT(*) FROM pedigree WHERE horse_id IN (SELECT horse_id FROM results)").fetchone()[0]
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
    行は「同コース」(競馬場・芝ダ・内外回り・距離)と「同じ A〜E」(競馬場・芝ダ・柵)。分からない行は出さない。
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
    inout = pd.Series([inner_outer(venue, ct, c) for c in df["course"]], index=df.index, dtype=object)
    rows = []
    distance = pd.to_numeric(race.get("distance"), errors="coerce")
    if distance == distance:
        rows.append(("同コース", df[(df["distance"] == distance) & (inout == inner_outer(venue, ct, race.get("course")))]))
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

# ---------------------------------------------------------------- 総合点(各ファクターを点数にして足す)
SCORE_EXCLUDE = {"popularity", "win_odds"}  # 市場の値は、ファクターの点数には入れない
SCORE_WEIGHTS: dict[str, float] = {}  # 列ごとの重み。書いていない列は 1
SCORE_MARKS = ["◎", "○", "▲"]


def score_columns() -> list[tuple[str, str]]:
    """点数にする列(良い向きのある SUMMARY の列)。"""
    return [(c, label) for c, label in SUMMARY["cols"]
            if c not in SCORE_EXCLUDE and COLUMN_INFO.get(c, (None, None))[1] is not None]


def column_points(values: pd.Series, good: str) -> pd.Series:
    """1つの列を、同じレースの中で 0〜10 点にする(いちばん良い馬が10点、悪い馬が0点。同じ値は平均の順位)。

    値がない馬、全馬が同じ値の列は、差がつかないよう真ん中の5点にする。
    """
    score = _good_score(values, good)
    known = score.dropna()
    if known.nunique() < 2:
        return pd.Series(5.0, index=values.index)
    rank = score.rank(method="average", ascending=False)  # 1が最良
    pts = (len(known) - rank) / (len(known) - 1) * 10
    return pts.fillna(5.0)


def score_race(one: pd.DataFrame) -> pd.DataFrame:
    """1レースの出走馬(factor_values の行)に、ファクターごとの点数・総合点(100点満点)・順位・印を付けて返す。

    列は pt_<列名>(0〜10点)、score(総合点)、score_rank(1が最上位)、score_mark(◎○▲△。9頭立て以下は△1頭)。
    列の順に1つずつ点数をつけて足し、最後に総合点の高い順に印をつける。
    """
    cols = [(c, label) for c, label in score_columns() if c in one]
    out = pd.DataFrame(index=one.index)
    total = pd.Series(0.0, index=one.index)
    wsum = 0.0
    for c, _ in cols:
        w = SCORE_WEIGHTS.get(c, 1.0)
        out[f"pt_{c}"] = column_points(one[c], COLUMN_INFO[c][1])
        total += out[f"pt_{c}"] * w
        wsum += w
    out["score"] = total / (10 * wsum) * 100 if wsum else np.nan
    out["score_rank"] = out["score"].rank(method="min", ascending=False)
    out["score_mark"] = ""
    n_marked = len(SCORE_MARKS) + (2 if len(one) >= 10 else 1)
    top = out["score"].sort_values(ascending=False, kind="stable").index[:n_marked]
    out.loc[top, "score_mark"] = (SCORE_MARKS + ["△", "△"])[:len(top)]
    return out


def score_day(values: pd.DataFrame) -> pd.DataFrame:
    """factor_values の全レースに score_race をかけて、同じ並びで返す。"""
    parts = [score_race(g) for _, g in values.groupby("race_id", sort=False)]
    return pd.concat(parts).reindex(values.index) if parts else pd.DataFrame(index=values.index)


# ---------------------------------------------------------------- 予想の要素別の点数と見解(予想ページ用)
# 要素ごとのモデルの勝率(p_win_*)と、画面での名前。factors.COLUMN と同じ並び
MODEL_FACTORS: dict[str, str] = {"p_win_quality": "馬の質", "p_win_fit": "適性", "p_win_rotation": "ローテ",
                                 "p_win_human": "騎手・厩舎", "p_win_track": "展開・馬場",
                                 "p_win_pedigree": "血統", "p_win_training": "調教"}
STRONG_PT, WEAK_PT = 8.0, 2.0  # 見解で「強み」「弱み」と書く点数


def model_factor_points(one: pd.DataFrame) -> pd.DataFrame:
    """1レースの予想の表から、要素ごとの勝率を同じレースの中で 0〜10 点にした表を返す。

    列は pt_<p_win_*>(いちばん高い馬が10点、低い馬が0点)と factor_total(点数の平均を100点満点にしたもの)。
    値のない要素(血統・調教を取得していないなど)は出さない。
    """
    out = pd.DataFrame(index=one.index)
    for c in MODEL_FACTORS:
        if c in one and one[c].notna().any():
            out[f"pt_{c}"] = column_points(one[c], "high")
    pts = out[[c for c in out if c.startswith("pt_")]]
    out["factor_total"] = pts.mean(axis=1) * 10 if len(pts.columns) else np.nan
    return out


def _rank_text(vals: pd.DataFrame, col: str, i) -> str:
    """列 col での、i 行目の馬のメンバー中の順位(「メンバー中2位」)。良い向きがなかったり、値がなければ空。"""
    good = COLUMN_INFO.get(col, (None, None))[1]
    if good is None or col not in vals:
        return ""
    score = _good_score(vals[col], good)
    if pd.isna(score.get(i)) or score.dropna().nunique() < 2:
        return ""
    return f"メンバー中{int(score.rank(method='min', ascending=False)[i])}位"


def _num(v, fmt: str = "{:.1f}") -> str:
    return fmt.format(v) if isinstance(v, (int, float, np.floating)) and v == v else "—"


def _has(v) -> bool:
    return v is not None and v == v and v != "" and v != "—"


def _quality_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    parts = []
    if _has(r.get("last5")):
        extra = [t for t in (f"平均着順{_num(r.get('avg_finish_5'))}", _rank_text(v, "avg_finish_5", i)) if t and "—" not in t]
        parts.append(f"近5走は{r['last5']}" + (f"({'・'.join(extra)})" if extra else ""))
    if _has(r.get("time_diff_5")):
        parts.append(f"勝ち馬とのタイム差は平均{_num(r['time_diff_5'], '{:.2f}')}秒({_rank_text(v, 'time_diff_5', i) or '順位なし'})")
    if _has(r.get("l3f_rank_3")):
        parts.append(f"上がりは頭数比{_num(r['l3f_rank_3'], '{:.2f}')}の順位({_rank_text(v, 'l3f_rank_3', i) or '順位なし'})")
    cc = r.get("class_change")
    if _has(cc) and cc != 0:
        parts.append("前走から格下げで相手が楽になる" if cc < 0 else "前走から昇級で相手が強くなる")
    return "。".join(parts) + "。" if parts else "過去の出走が少なく、近走の材料がありません。"


def _fit_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    names = [("rec_ct", "芝/ダ"), ("rec_dist", "距離帯"), ("rec_dir", "回り"), ("rec_venue", "競馬場"), ("rec_ground", "馬場")]
    got = [f"{n} {r[c]}" for c, n in names if _has(r.get(c))]
    if not got:
        return "同じ条件の成績がありません。"
    text = "今回と同じ条件の成績(1着-2着-3着-着外)は、" + "・".join(got) + "。"
    top3 = lambda c: sum(int(x) for x in str(r.get(c, "0-0-0-0")).split("-")[:3]) if _has(r.get(c)) else 0
    text += "近い距離で3着内の実績があります。" if top3("rec_dist") else "近い距離での3着内の実績はありません。"
    return text


def _rotation_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    if not _has(r.get("rotation")):
        return "前走の情報がありません(初出走など)。"
    text = f"{r['rotation']}"
    if _has(r.get("days_since_last")):
        text += f"(前走から{int(r['days_since_last'])}日)"
    return text + "。"


def _human_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    parts = []
    if _has(r.get("jockey")):
        rank = f"今年の勝利数{int(r['jockey_wins_year'])}勝・リーディング{int(r['jockey_rank'])}位" if _has(r.get("jockey_rank")) else ""
        parts.append(f"{r['jockey']}騎手" + (f"({rank})" if rank else ""))
    for col, label in (("jockey_ct_top3_rate", "同じ芝/ダでの3着内率"), ("jockey_venue_top3_rate", "この競馬場での3着内率")):
        if _has(r.get(col)):
            parts.append(f"{label}{r[col] * 100:.0f}%({_rank_text(v, col, i) or '順位なし'})")
    if _has(r.get("pair_n")) and r["pair_n"] > 0:
        parts.append(f"この馬とは{int(r['pair_n'])}回で3着内率{_num(r.get('pair_top3_rate', np.nan) * 100, '{:.0f}')}%")
    elif _has(r.get("pair_n")):
        parts.append("この馬とは初コンビ")
    if r.get("jockey_change") == "乗り替わり":
        parts.append("前走から乗り替わり")
    return "。".join(parts) + "。" if parts else "騎手の情報がありません。"


def _track_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    parts = []
    if _has(r.get("style")):
        parts.append(f"脚質は{r['style']}と推定")
    if _has(r.get("pace_forecast")):
        parts.append(f"このメンバーのペース予想は{r['pace_forecast']}")
    fit = r.get("pace_fit")
    if _has(fit):
        parts.append("展開は向く" if fit > 0.1 else "展開は向きにくい" if fit < -0.1 else "展開による有利不利は小さい")
    if _has(r.get("number")) and _has(r.get("inner_pos")):
        pos = "内" if r["inner_pos"] <= 1 / 3 else "外" if r["inner_pos"] > 2 / 3 else "中"
        parts.append(f"{int(r['number'])}番は{pos}寄りの枠")
    return "。".join(parts) + "。" if parts else "展開・枠の材料がありません。"


def _pedigree_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    if not _has(r.get("sire")):
        return "血統のデータを取得していないため、評価に入れていません。"
    parts = [f"父{r['sire']}" + (f"・母の父{r['damsire']}" if _has(r.get("damsire")) else "")]
    for col, label in (("sire_ctdist_top3_rate", "父の産駒の同じ芝ダ×距離帯での3着内率"),
                       ("damsire_ctdist_top3_rate", "母の父の産駒の同じ条件での3着内率")):
        if _has(r.get(col)):
            parts.append(f"{label}{r[col] * 100:.0f}%({_rank_text(v, col, i) or '順位なし'})")
    return "。".join(parts) + "。"


def _training_text(v: pd.DataFrame, i) -> str:
    r = v.loc[i]
    parts = []
    if _has(r.get("train_last_place")):
        parts.append(f"最後の追い切りは{r['train_last_place']}")
    if _has(r.get("jv_last_f4")):
        z = r.get("jv_last_f4_z")
        tail = f"(同じコースの標準より{abs(z):.1f}秒{'速い' if z < 0 else '遅い'})" if _has(z) else ""
        parts.append(f"4Fは{r['jv_last_f4']:.1f}秒{tail}")
    if _has(r.get("train_critic")) or _has(r.get("train_rank")):
        parts.append("短評「" + str(r.get("train_critic") or "—") + "」・評価" + str(r.get("train_rank") or "—"))
    if _has(r.get("train_day")):  # 調教の詳細が取れた馬
        parts.append(f"{r['train_day']}の追い切りで、騎乗は{r['train_rider'] if _has(r.get('train_rider')) else '不明'}"
                     + (f"・脚色は{r['train_load']}" if _has(r.get("train_load")) else ""))
        parts.append(f"併せ馬は{r['train_partner']}" if _has(r.get("train_partner")) else "併せ馬の記載はなし(単走)")
    elif parts:
        parts.append("騎乗者と併せ馬は取得できていません(netkeiba の詳細が会員限定のため)")
    return "。".join(parts) + "。" if parts else "追い切りのデータを取得していないため、評価に入れていません。"


# 要素ごとの根拠の書き方(MODEL_FACTORS の並び)
EVIDENCE = {"p_win_quality": _quality_text, "p_win_fit": _fit_text, "p_win_rotation": _rotation_text,
            "p_win_human": _human_text, "p_win_track": _track_text, "p_win_pedigree": _pedigree_text,
            "p_win_training": _training_text}


def grade_label(pt: float) -> str:
    return ("高評価" if pt >= STRONG_PT else "やや高評価" if pt >= 6 else "平均的" if pt >= 4
            else "やや低評価" if pt > WEAK_PT else "低評価")


def race_reviews(one: pd.DataFrame, points: pd.DataFrame, values: pd.DataFrame | None = None) -> list[dict]:
    """全頭の見解。印のついた馬(◎○▲△☆★)が先で、印のない馬は総合スコアの高い順。

    one は印・勝率つきの1レース分、points は model_factor_points、values は factor_values の同じレース分(なくてもよい)。
    各馬: {title(印・馬番・馬名), sub(総合スコア・人気・オッズ), summary(勝率と印の理由), rows(要素ごとの
    {name, pt, grade, text}。text は values があるとき、その要素の根拠の文)}。
    """
    if "mark" not in one:
        return []
    order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    rows = one.copy()
    rows["_o"] = rows["mark"].map(order).fillna(5)
    rows["_total"] = -points["factor_total"].reindex(rows.index).fillna(-1) if "factor_total" in points else 0
    vals = None
    if values is not None and len(values):
        vals = values.drop_duplicates("number").set_index("number", drop=False)
    out = []
    for i, r in rows.sort_values(["_o", "_total", "number"]).iterrows():
        label = f"{r['mark']}{' ★' if r.get('star') == '★' else ''}".strip()
        sub = []
        total = points["factor_total"].get(i) if "factor_total" in points else None
        if total is not None and total == total:
            sub.append(f"総合スコア{total:.1f}/100")
        if r["popularity"] == r["popularity"]:
            sub.append(f"{int(r['popularity'])}番人気・{r['win_odds']}倍")
        parts = []
        pa, pm = r.get("p_win_ability"), r.get("p_win_market")
        if pa == pa and pm == pm and pa is not None and pm is not None:
            parts.append(f"勝率は能力で{pa * 100:.1f}%、オッズ込みで{pm * 100:.1f}%。")
            if pm >= pa * 1.5 and pm - pa >= 0.03:
                parts.append("人気が勝率を大きく押し上げていて、能力だけなら評価は下がります。")
            elif pa >= pm * 1.3 and pa - pm >= 0.02:
                parts.append("能力はオッズ(人気)の評価より高く、妙味があります。")
        elif pa == pa and pa is not None:
            parts.append(f"能力だけの勝率は{pa * 100:.1f}%(オッズ未取得)。")
        got = {MODEL_FACTORS[c]: points.loc[i, f"pt_{c}"] for c in MODEL_FACTORS if f"pt_{c}" in points}
        strong = [f"{k}{v:.0f}/10" for k, v in sorted(got.items(), key=lambda kv: -kv[1]) if v >= STRONG_PT]
        weak = [f"{k}{v:.0f}/10" for k, v in sorted(got.items(), key=lambda kv: kv[1]) if v <= WEAK_PT]
        if strong:
            parts.append("強みは" + "・".join(strong) + "。")
        if weak:
            parts.append("弱みは" + "・".join(weak) + "。")
        if not strong and not weak and got:
            parts.append("どの要素も突出せず、全体のバランスで評価しています。")
        if r.get("mark") == "☆":
            ev, kind = (r["ev_market"], "オッズ込み") if pd.notna(r.get("ev_market")) else (r["ev_ability"], "能力")
            parts.append(f"{kind}の期待値が{ev:.2f}と1を超える穴馬です。")
        if r.get("star") == "★":
            parts.append(f"オッズ込みの期待値が{r['ev_market']:.2f}で、オッズより高く評価されています(買えば得になる印ではありません)。")
        factors = []
        for c, name in MODEL_FACTORS.items():
            if f"pt_{c}" not in points:
                continue
            pt = float(points.loc[i, f"pt_{c}"])
            text = ""
            if vals is not None and int(r["number"]) in vals.index:
                text = EVIDENCE[c](vals, int(r["number"]))
            factors.append({"name": name, "pt": pt, "grade": grade_label(pt), "text": text})
        out.append({"mark": label, "number": int(r["number"]), "name": r["horse_name"], "sub": " ・ ".join(sub),
                    "summary": "".join(parts), "rows": factors})
    return out


# ---------------------------------------------------------------- レース展開
def course_history(conn: sqlite3.Connection, race, before: str, days: int = 5) -> pd.DataFrame:
    """同コース(同じ競馬場の、芝ダ・内回り外回り・距離が同じコース)の、before より前の直近 days 開催日のレースごとの展開。新しい順。

    列: 日付・レース・馬場・前半3F・後半3F・ペース差・ペース・先行有利度(前に付けた馬の3着内率 − ほかの馬)・
    勝ち馬の位置(最初のコーナーの位置÷頭数)。コースが分からなければ空。
    """
    venue, ct = race.get("venue"), race.get("course_type")
    distance = pd.to_numeric(race.get("distance"), errors="coerce")
    if not isinstance(venue, str) or not isinstance(ct, str) or distance != distance:
        return pd.DataFrame()
    inout = inner_outer(venue, ct, race.get("course"))
    ids = [rid for rid, date, course in conn.execute(
        "SELECT race_id, date, course FROM races WHERE venue = ? AND course_type = ? AND distance = ? AND date < ? "
        "ORDER BY date DESC", (venue, ct, int(distance), before)) if inner_outer(venue, ct, course) == inout]
    if not ids:
        return pd.DataFrame()
    q = ",".join("?" * len(ids))
    df = pd.read_sql_query(
        f"SELECT c.race_id, c.date, c.round, c.name, c.ground, c.pace_first3f, c.pace_last3f, r.finish, r.passing, "
        f"r.number FROM races c JOIN results r USING (race_id) WHERE c.race_id IN ({q}) AND r.finish IS NOT NULL",
        conn, params=ids)
    df = df[df["date"].isin(sorted(df["date"].unique(), reverse=True)[:days])]
    if df.empty:
        return pd.DataFrame()
    return _flow_rows(df).sort_values(["日付", "R"], ascending=False).reset_index(drop=True)


def _flow_rows(df: pd.DataFrame) -> pd.DataFrame:
    """race_id・date・round・name・ground・pace_first3f・pace_last3f・finish・passing・number の表から、
    レースごとの展開(ペース差・先行有利度・内有利度・勝ち馬)を1行ずつにする。"""
    from .features import parse_early_position
    df = df.copy()
    df["field"] = df.groupby("race_id")["number"].transform("count")
    df["early"] = df["passing"].map(parse_early_position) / df["field"]
    rows = []
    for rid, g in df.groupby("race_id", sort=False):
        f = g[g["early"].notna()]
        front = f["early"] <= 1 / 3
        top3 = (f["finish"] <= 3).astype(float)
        win = g[g["finish"] == 1]
        pos = g["number"] / g["field"]
        inner, outer = pos <= 1 / 3, pos >= 2 / 3
        top3_all = (g["finish"] <= 3).astype(float)
        first, last = pd.to_numeric(g["pace_first3f"].iloc[0], errors="coerce"), pd.to_numeric(g["pace_last3f"].iloc[0], errors="coerce")
        rows.append({"race_id": rid, "日付": g["date"].iloc[0], "R": int(g["round"].iloc[0]), "レース": g["name"].iloc[0],
                     "馬場": g["ground"].iloc[0], "前半3F": first, "後半3F": last, "ペース差": first - last,
                     "ペース": pace_label(first - last),
                     "先行有利度": top3[front].mean() - top3[~front].mean() if front.any() and (~front).any() else np.nan,
                     "内有利度": top3_all[inner].mean() - top3_all[outer].mean() if inner.any() and outer.any() else np.nan,
                     "勝ち馬": int(win["number"].iloc[0]) if len(win) else np.nan,
                     "勝ち馬の位置": win["early"].iloc[0] if len(win) and win["early"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def same_day_history(conn: sqlite3.Connection, race, date: str) -> pd.DataFrame:
    """同じ日・同じ競馬場・同じ芝ダで、このレースより前の R の展開(R の若い順)。結果が DB にあるレースだけ。

    芝の状態や内外の有利不利は、後のレースほど変わる。メインの 11R を見るとき、同じ日の前のレースが手がかりになる。
    """
    venue, ct, rnd = race.get("venue"), race.get("course_type"), pd.to_numeric(race.get("round"), errors="coerce")
    if not isinstance(venue, str) or not isinstance(ct, str) or rnd != rnd:
        return pd.DataFrame()
    df = pd.read_sql_query(
        "SELECT c.race_id, c.date, c.round, c.name, c.ground, c.pace_first3f, c.pace_last3f, r.finish, r.passing, "
        "r.number FROM races c JOIN results r USING (race_id) WHERE c.date = ? AND c.venue = ? AND c.course_type = ? "
        "AND c.round < ? AND r.finish IS NOT NULL", conn, params=(date, venue, ct, int(rnd)))
    if df.empty:
        return pd.DataFrame()
    return _flow_rows(df).sort_values("R").reset_index(drop=True)


def same_day_text(venue: str, course_type: str, today: pd.DataFrame) -> str:
    """同じ日の前のレースの見立て。前・内の有利度は、ここまでのレースの平均で、件数が少ないうちは目安。"""
    if today.empty:
        return ""
    n = len(today)
    diff, front, inner = today["ペース差"].mean(), today["先行有利度"].mean(), today["内有利度"].mean()
    parts = [f"同じ日の{venue}{course_type}は、ここまで{n}レース。"]
    if diff == diff:
        parts.append(f"ペース差は平均 {diff:+.1f}秒で{pace_label(diff)}。")
    for name, v, hi, lo in (("前", front, "前に付けた馬", "後ろの馬"), ("内", inner, "内の馬", "外の馬")):
        if v == v:
            parts.append(f"{hi}の3着内率は{lo}より {v * 100:+.0f}pt" + ("(" + (f"{name}有利" if v >= 0.1 else f"{'差し' if name == '前' else '外'}有利" if v <= -0.1 else "差は小さい") + ")") + "。")
    if n < 4:
        parts.append("レース数が少ないので、目安です。")
    return "".join(parts)


def race_flow(one: pd.DataFrame, history: pd.DataFrame) -> dict:
    """1レースの展開の見立て。one は factor_values の1レース分、history は course_history の結果。

    lineup: 脚質ごとの馬番(逃げ・先行・差し・追込・マクリ・不明)。forecast: メンバーの脚質構成からのペース予想。
    course_pace: 同コース直近のペース差の平均と呼び名。edge: 前に付ける馬の有利度(−1〜+1。front_edge)。
    favored: その結果、有利な脚質の言葉。text: 画面に出す1〜2行の説明。
    """
    lineup = {k: [int(n) for n in one.loc[one["style"] == k, "number"]] for k in STYLES + ["マクリ", "不明"]}
    forecast = one["pace_forecast"].iloc[0] if len(one) else "不明"
    diff = history["ペース差"].mean() if not history.empty else np.nan
    cedge = history["先行有利度"].mean() if not history.empty else np.nan
    edge = front_edge(forecast, diff, cedge)
    favored = "逃げ・先行が有利" if edge >= 0.25 else "差し・追込が有利" if edge <= -0.25 else "脚質による差は小さい"
    parts = [f"このメンバーは{forecast}(逃げ{len(lineup['逃げ'])}頭・先行{len(lineup['先行'])}頭)。"]
    if diff == diff:
        parts.append(f"同コース直近{history['ペース差'].notna().sum()}レースは平均ペース差 {diff:+.1f}秒で{pace_label(diff)}"
                     + ("" if cedge != cedge else f"、前に付けた馬の3着内率はほかより {cedge * 100:+.0f}pt") + "。")
    parts.append(f"見立て: {favored}。")
    return {"lineup": lineup, "forecast": forecast, "course_pace": (diff, pace_label(diff)), "course_edge": cedge,
            "edge": edge, "favored": favored, "text": "".join(parts)}
