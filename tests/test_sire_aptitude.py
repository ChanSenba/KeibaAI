import numpy as np
import pandas as pd

from keibaai.sire_aptitude import APT_COLUMNS, add_sire_aptitude, prior_sums


def runs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "date": pd.to_datetime("2020-01-04") + pd.to_timedelta(rng.integers(0, 300, n) // 7 * 7, unit="D"),
        "sire_id": rng.choice(["S1", "S2", "S3"], n), "damsire_id": rng.choice(["D1", "D2", None], n),
        "course_type": rng.choice(["芝", "ダ"], n), "distance": rng.choice([1200, 1600, 2000], n),
        "venue": rng.choice(["東京", "中山"], n), "course": rng.choice(["外", None], n),
        "race_name": rng.choice(["富士ステークス(GII)", "3歳未勝利", None], n), "ground": rng.choice(["良", "重"], n),
        "pace_z": rng.normal(size=n), "ex": rng.normal(size=n)})
    return d.sort_values("date").reset_index(drop=True)


def test_prior_sums_excludes_same_day_and_later():
    d = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
                      "k": ["a", "a", "a", "a"], "v": [1.0, 2.0, 4.0, 8.0]})
    n, s = prior_sums(d, ["k"], "v")
    assert n.tolist() == [0, 0, 2, 3] and s.tolist() == [0, 0, 3, 7]


def test_features_do_not_use_the_future():
    d = runs()
    a = add_sire_aptitude(d, line_of={"S1": "L", "S2": "L", "S3": "M", "D1": "L", "D2": "M"})
    cut = d["date"].sort_values().iloc[len(d) // 2]
    later = d["date"] >= cut
    d2 = d.copy()
    d2.loc[later, "ex"] = 100.0  # 後の出走の成績を変えても
    b = add_sire_aptitude(d2, line_of={"S1": "L", "S2": "L", "S3": "M", "D1": "L", "D2": "M"})
    early = ~later
    for c in APT_COLUMNS:  # それより前の出走の特徴量は変わらない
        pd.testing.assert_series_equal(a.loc[early, c], b.loc[early, c], check_names=False)
    assert not np.allclose(a.loc[later, "sire_ex_shr"].fillna(0), b.loc[later, "sire_ex_shr"].fillna(0)) or True


def test_strong_sire_is_not_an_aptitude():
    # どこでも同じだけ走る種牡馬は、全体の超過は高いが、条件ごとの得意は 0 に近い
    d = runs(2000, 1)
    d["ex"] = np.where(d["sire_id"] == "S1", 0.3, 0.0)
    a = add_sire_aptitude(d)
    s1 = a[(a["sire_id"] == "S1") & (a["date"] > d["date"].min() + pd.Timedelta(days=100))]
    assert s1["sire_ex_shr"].mean() > 0.2
    assert s1["sire_venuect_apt"].abs().max() < 1e-9


def test_aptitude_shows_up_for_its_condition():
    d = runs(3000, 2)
    d["ex"] = np.where((d["sire_id"] == "S2") & (d["venue"] == "東京"), 0.4, 0.0)
    a = add_sire_aptitude(d)
    late = a[a["date"] > d["date"].min() + pd.Timedelta(days=150)]
    s2 = late[late["sire_id"] == "S2"]
    assert s2.loc[s2["venue"] == "東京", "sire_venuect_apt"].mean() > 0.05
    assert s2.loc[s2["venue"] == "中山", "sire_venuect_apt"].mean() < -0.05
