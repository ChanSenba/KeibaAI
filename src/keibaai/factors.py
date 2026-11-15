"""要素(馬の質・適性・ローテーション・騎手と厩舎・展開と馬場・血統・調教)の定義。

予想のニューラルネット(nn_model)は、要素ごとの小さい網の点数を足して勝率にする。どの特徴量をどの要素に入れるかと、
予想の表での列名をここで決める。
"""
from __future__ import annotations

from .features import (APTITUDE_FEATURES, FIGURE_FEATURES, FORM_FEATURES, HUMAN_EXTRA, JOCKEY_DELTA_FEATURES,
                       BIAS_FEATURES, COURSE_FLOW_FEATURES, LAP_FEATURES, OTHER_FEATURES, PACE_FEATURES,
                       PAST_SCORE_FEATURES, PEDIGREE_FEATURES, RACE_LEVEL_FEATURES, RELATIVE_FEATURES,
                       ROTATION_FEATURES, TRAINING_FEATURES)

FACTORS: dict[str, list[str]] = {
    "馬の質": ["age", "weight_carried", "body_weight", "body_weight_diff", "n_prev", "prev_finish",
              "avg_finish_3", "avg_finish_5", "win_rate", "top3_rate", "avg_last3f_3",
              "race_class", "class_change", "distance", "course_type", "ground", *FORM_FEATURES,
              *FIGURE_FEATURES, *RELATIVE_FEATURES, *PAST_SCORE_FEATURES, *RACE_LEVEL_FEATURES],
    "適性": ["distance", "course_type", "venue", "ground", "distance_change", "same_dist_top3_rate", *APTITUDE_FEATURES],
    "ローテーション": ["age", "n_prev", "days_since_last", *ROTATION_FEATURES, *OTHER_FEATURES],
    "騎手・厩舎": ["jockey_changed", "jockey_n", "jockey_win_rate", "jockey_top3_rate", "trainer_n", "trainer_top3_rate",
               *HUMAN_EXTRA, *JOCKEY_DELTA_FEATURES],
    "展開・馬場": ["number", "bracket", "field_size", "distance", "course_type", "venue", "ground", "early_pos_rel_3",
                 *BIAS_FEATURES, *PACE_FEATURES, *LAP_FEATURES, *COURSE_FLOW_FEATURES],
    "血統": ["distance", "course_type", "venue", "ground", *PEDIGREE_FEATURES],
    "調教": ["age", "n_prev", "days_since_last", "race_class", *TRAINING_FEATURES],
}
COLUMN = {"馬の質": "p_win_quality", "適性": "p_win_fit", "ローテーション": "p_win_rotation",
          "騎手・厩舎": "p_win_human", "展開・馬場": "p_win_track",
          "血統": "p_win_pedigree", "調教": "p_win_training"}
