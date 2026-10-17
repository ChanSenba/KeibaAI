"""本番との整合性: 中止、未来追加、予測日の欠損、同件数での訂正。"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from keibaai import db, predict, blend, model
from keibaai.features import (build_features, load_results, result_mask, _track_bias,
                              ABILITY_MODEL_FEATURES, MARKET_MODEL_FEATURES)
from keibaai.jvtraining import add_course_relative, add_jv_features, JV_FEATURES
from test_model import synthetic_db
from test_entries import _entries_for_future


def test_nonfinishers_count_but_never_become_entries(monkeypatch):
    conn = synthetic_db(n_days=6, races_per_day=2)
    rid, num, horse = conn.execute("SELECT race_id,number,horse_id FROM results ORDER BY race_id,number LIMIT 1").fetchone()
    conn.execute("UPDATE results SET finish=NULL,status='中' WHERE race_id=? AND number=?", (rid,num))
    raw = load_results(conn)
    df = build_features(raw)
    r = df[(df.race_id==rid) & (df.number==num)].iloc[0]
    assert r.field_size == 10 and r.win == 0 and r.top3 == 0 and not r.is_entry
    assert result_mask(df).all()
    assert len(df) == 120
    h = df[df.horse_id==horse].sort_values('date')
    if len(h)>1:
        assert h.iloc[1].n_prev == 1
    _entries_for_future(conn, '2030-01-05')
    seen = []
    real = predict._fit
    def fit(train, *args):
        seen.append(train)
        return real(train, *args)
    monkeypatch.setattr(predict, '_fit', fit)
    out = predict.predict_day(conn, '2030-01-05', rounds=5)
    assert len(out)==10 and out.race_id.nunique()==1
    assert all(len(t)==120 for t in seen)
    assert np.isclose(out.p_win_ability.sum(),1)


@pytest.mark.parametrize('status', ['取','除'])
def test_nonstarters_are_not_in_field_or_history(status):
    raw=load_results(synthetic_db(n_days=3,races_per_day=2))
    rid=raw.race_id.iloc[0]
    raw.loc[0,['finish','status']]=[np.nan,status]
    df=build_features(raw)
    assert len(df)==len(raw)-1
    assert df.loc[df.race_id==rid,'field_size'].eq(9).all()


def test_dnf_is_in_rate_denominators():
    from keibaai.features import _expanding_rate
    f=pd.DataFrame({'date':pd.to_datetime(['2026-01-01']*2+['2026-01-02']),
                    'j':['a']*3,'finish':[1,np.nan,np.nan], 'status':[None,'中',None],
                    'is_entry':[False,False,True],'win':[1,0,0]})
    n,r=_expanding_rate(f,'j','win')
    assert n.iloc[-1]==2 and r.iloc[-1]==.5


def test_legacy_demoted_finish_is_recovered_without_db_write():
    conn=synthetic_db(n_days=2,races_per_day=2)
    conn.execute("UPDATE results SET finish=NULL,status='3(降)' WHERE rowid=1")
    conn.execute("UPDATE results SET finish=NULL,status='中' WHERE rowid=2")
    df=load_results(conn)
    assert df.loc[df.status=='3(降)','finish'].iloc[0]==3
    assert conn.execute('SELECT finish FROM results WHERE rowid=1').fetchone()[0] is None
    features=build_features(df)
    assert features.loc[features.status=='中','win'].iloc[0]==0
    assert features.loc[features.status=='3(降)','top3'].iloc[0]==1


def test_all_features_are_invariant_to_future_laps():
    raw=load_results(synthetic_db(n_days=10,races_per_day=2))
    raw['pace_first3f']=raw.date.rank(method='dense')*.3+34
    raw['pace_last3f']=35.
    cut=sorted(raw.date.unique())[6]
    full=build_features(raw)
    prefix=build_features(raw[raw.date<=cut].copy())
    changed=raw.copy()
    changed.loc[changed.date>cut,'pace_first3f']=80.
    changed=build_features(changed)
    cols=list(dict.fromkeys(ABILITY_MODEL_FEATURES+MARKET_MODEL_FEATURES))
    for other in (prefix,changed[changed.date<=cut]):
        a=full[full.date<=cut].set_index(['race_id','number'],drop=False)[cols]
        b=other.set_index(['race_id','number'],drop=False)[cols].reindex(a.index)
        pd.testing.assert_frame_equal(a,b,check_categorical=False)


def test_bias_predict_date_and_ten_day_boundary():
    rows=[]
    for day,entry in [('2026-01-01',False),('2026-01-11',True),('2026-01-12',True)]:
        for n in range(1,7):
            rows.append(dict(date=pd.Timestamp(day),venue='東京',course_type='芝',number=n,
                             field_size=6,early_pos=n/6,finish=np.nan if entry else n,is_entry=entry))
    f=pd.DataFrame(rows)
    b=_track_bias(f).set_index('date')
    assert b.loc['2026-01-01','track_inner_bias']!=b.loc['2026-01-01','track_inner_bias']
    assert b.loc['2026-01-11','track_inner_bias']==1.
    assert pd.isna(b.loc['2026-01-12','track_inner_bias'])
    assert _track_bias(f.assign(is_entry=True,finish=np.nan)).track_inner_bias.isna().all()


def training_rows():
    return pd.DataFrame({'date':['2026-01-01']*30+['2026-01-02']*2+['2026-01-03'],
        'kind':['slope']*33,'center':['0']*33,'course':[None]*33,
        'f4':[52.]*15+[54.]*15+[52.,56.,50.], 'f3':[39.]*33,'f2':[26.]*33,'l1':[13.]*33})


def test_training_prior_only_ties_and_minimum():
    tr=training_rows()
    a=add_course_relative(tr)
    assert a.iloc[:30].f4_z.isna().all()
    assert a.iloc[30].f4_z==-1. and a.iloc[30].f4_pct==.25
    assert a.iloc[31].f4_z==3. and a.iloc[31].f4_pct==1.
    prefix=add_course_relative(tr.iloc[:32])
    pd.testing.assert_frame_equal(a.iloc[:32],prefix)
    shuffled=add_course_relative(tr.sample(frac=1,random_state=1))
    pd.testing.assert_frame_equal(a,shuffled.sort_index())


def test_future_training_does_not_change_race_features():
    conn=db.connect(':memory:')
    tr=training_rows().assign(horse_id='h',time='0600')
    # 30頭の基準で複合キーの重複を避ける。
    tr.loc[:29,'horse_id']=[f'base{i}' for i in range(30)]
    tr.loc[31,'time']='0700'
    tr.iloc[:32].to_sql('jv_training',conn,if_exists='append',index=False)
    target=pd.DataFrame({'race_id':['r'],'horse_id':['h'],'date':['2026-01-03']})
    a=add_jv_features(conn,target.copy())
    tr.iloc[32:].to_sql('jv_training',conn,if_exists='append',index=False)
    b=add_jv_features(conn,target.copy())
    pd.testing.assert_frame_equal(a[JV_FEATURES],b[JV_FEATURES])


@pytest.mark.parametrize('table,insert,update', [
    ('pedigree',"INSERT INTO pedigree(horse_id,sire_id) VALUES('h','s')","UPDATE pedigree SET sire_id='new'"),
    ('training',"INSERT INTO training(race_id,number,seq,rank) VALUES('r',1,0,'B')","UPDATE training SET rank='A'"),
    ('results',"INSERT INTO results(race_id,number,finish,win_odds) VALUES('r',1,1,2)","UPDATE results SET win_odds=3"),
    ('races',"INSERT INTO races(race_id,date,ground) VALUES('r','2026-01-01','良')","UPDATE races SET ground='重'"),
    ('jv_training',"INSERT INTO jv_training(horse_id,date,time,kind,f4) VALUES('h','2026-01-01','0600','slope',52)","UPDATE jv_training SET f4=53"),
])
def test_cache_content_updates_invalidate_even_across_connections(tmp_path,table,insert,update):
    path=tmp_path/'db.sqlite'
    conn=db.connect(path)
    conn.execute(insert)
    conn.commit()
    calls=[]
    def compute():
        calls.append(1)
        return len(calls)
    def get():
        return predict._cached(conn,'test','2026-02-01',False,(),compute)
    assert get()==get()==1
    other=sqlite3.connect(path)  # db.connect を使わない訂正にもトリガーが効く
    other.execute(update)
    other.commit()
    assert get()==2
    other.execute(update)  # 同じ内容の書き直しでは更新番号を進めない
    other.commit()
    assert get()==2
    other.execute(f'DELETE FROM {table}')
    other.commit()
    assert get()==3
    other.close()
    conn.close()


@pytest.mark.parametrize('use_market',[False,True])
def test_blend_passes_model_selection_to_both_periods(monkeypatch,use_market):
    calls=[]
    f=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2025-01-01','2026-01-01'])})
    def evaluate(frame,cut,use_market):
        calls.append((frame.date.max(),cut,use_market))
        return None,frame
    monkeypatch.setattr(blend,'evaluate',evaluate)
    monkeypatch.setattr(blend,'fit_blend',lambda _: object())
    monkeypatch.setattr(blend,'apply_blend',lambda frame,b:frame)
    blend.predict_test_blend(f,'2026-01-01',use_market=use_market)
    assert [x[2] for x in calls]==[use_market,use_market]
    assert calls[0][0]<pd.Timestamp('2026-01-01')
    assert calls[0][1]=='2025-01-01'


@pytest.mark.parametrize('use_market',[False,True])
def test_cli_and_gui_forward_blend_selection(monkeypatch,tmp_path,capsys,use_market):
    from keibaai import cli,gui,features
    f=pd.DataFrame({'race_id':['r']*2,'number':[1,2], 'date':pd.to_datetime(['2026-01-01']*2),
        'p_win':[.6,.4],'p_model':[.6,.4],'p_mkt':[.5,.5],'win':[1,0],
        'win_odds':[2.,3.],'popularity':[1,2]})
    seen=[]
    def mixed(df,cut,**kwargs):
        seen.append(kwargs['use_market'])
        return f,blend.Blend(.5,.5,-1.,-1.)
    monkeypatch.setattr(features,'load_results',lambda _:f)
    monkeypatch.setattr(features,'build_features',lambda df:df)
    monkeypatch.setattr(blend,'predict_test_blend',mixed)
    args=['backtest','--db',str(tmp_path/'k.db'),'--test-from','2026-01-01','--blend']
    cli.main(args+(['--market'] if use_market else []))
    text=capsys.readouterr().out
    assert ('モデルだけ(オッズ込)' if use_market else 'モデルだけ(能力)') in text
    gui.backtest_run(None,'2026-01-01',use_market,1.,None,100,blend=True)
    assert seen==[use_market,use_market]


def test_legacy_readonly_db_does_not_reuse_untracked_cache(tmp_path):
    path=tmp_path/'legacy.db'
    c=sqlite3.connect(path)
    c.executescript(db.SCHEMA)
    c.close()
    c=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    calls=[]
    for _ in range(2):
        predict._cached(c,'test','2026-01-01',False,(),lambda:calls.append(1))
    assert len(calls)==2 and not (tmp_path/predict.CACHE_DIR).exists()
    c.close()
    c=db.connect(path)  # 通常の起動では既存DBにも更新検知をインストール
    old=db.model_revision(c)
    c.execute("INSERT INTO races(race_id) VALUES ('r')")
    assert db.model_revision(c)!=old
    c.rollback()
    assert db.model_revision(c)==old
    c.close()


def test_constant_pace_has_no_finite_z_score():
    from keibaai.features import _past_pace_z
    df=pd.DataFrame({'race_id':[str(i) for i in range(100)],'date':pd.date_range('2026-01-01',periods=100),
                    'course_type':['芝']*100,'dist_bucket':[4]*100,'pace_first3f':[36.4]*100})
    assert _past_pace_z(df).isna().all()
