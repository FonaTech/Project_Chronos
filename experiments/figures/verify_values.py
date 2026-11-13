"""Independently compare plotted export records with the immutable JSON snapshot."""
from pathlib import Path
import json,math
import numpy as np

package=Path(__file__).resolve().parent
data=package/'source_data';out=package/'figures'
checks=[];point_count=0

def load(rel):return json.loads((data/rel).read_text())
def records(fig):return json.loads((out/'data'/(fig+'.json')).read_text())['series']
def same(label,actual,expected,tolerance=0):
    global point_count
    assert len(actual)==len(expected),(label,len(actual),len(expected))
    for a,b in zip(actual,expected):
        if b is None: assert a is None,(label,a,b)
        else: assert a is not None and abs(float(a)-float(b))<=tolerance,(label,a,b)
    point_count+=len(expected);checks.append(label)

for r in records('loop_depth_ce_curves'):
    d=load(r['source']);same(r['label']+' curve steps',r['x'],[e['step'] for e in d['evals']])
    same(r['label']+' curve CE',r['y'],[e['valid_ce'] for e in d['evals']])
    same(r['label']+' curve minutes',r['train_minutes'],[e['train_wall_s']/60 for e in d['evals']])
for r in records('loop_depth_efficiency'):
    d=load(r['source'])['final']; t=r['transform']
    v=d['step']/(d['train_wall_s']/60) if t.startswith('step') else d['peak_gb'] if t=='peak_gb' else d['tokens']/1e6
    same(r['label'],r['y'],[v])
for r in records('per_source_ce'):
    d=load(r['source'])['final']['per_source_ce'];same(r['label']+' per-corpus',r['y'],[d[s] for s in r['x']])
for r in records('per_source_ce_eq_step'):
    d=load(r['source']);p=min(d['evals'],key=lambda e:abs(e['step']-800))
    same(r['label']+' matched-step CE',r['y'],[p['valid_ce']])
    same(r['label']+' observed step',r['x'],[p['step']])
for r in records('block_vs_layer_ce'):
    if isinstance(r['source'],list):
        aa={e['step']:e['valid_ce'] for e in load(r['source'][0])['evals']}
        bb={e['step']:e['valid_ce'] for e in load(r['source'][1])['evals']}
        steps=sorted(set(aa)&set(bb));same('paired exact common steps',r['x'],steps)
        same('paired exact CE differences',r['y'],[aa[s]-bb[s] for s in steps])
    else:
        d=load(r['source']);same(r['label']+' order curve CE',r['y'],[e['valid_ce'] for e in d['evals']])
for r in records('dt_learned'):
    d=load(r['source']);group=int(r['label'].split()[-1]);cfg=d['config']
    raw=[e['dt'][group] for e in d['evals'] if len(e.get('dt',[]))>group and e['dt'][group] is not None]
    expected=cfg['recurrent_dt_min']+(cfg['recurrent_dt_max']-cfg['recurrent_dt_min'])/(1+np.exp(-np.array(raw)))
    same(r['label']+' stable inverse dt',r['y'],list(expected),tolerance=1e-12)
for r in records('residency_bench'):
    d=load(r['source']);same(r['label']+' summary',r['y'],[d['summary'][m][r['label']] for m in r['x']])
for r in records('cluster_locality_sweep'):
    d=load(r['source'])
    if r['label']=='lambda zero baseline':v=d['arms']['baseline']['lazy_decode']['window']['cluster_reads_per_token']
    elif r['label'].endswith('reads'):v=d['arms']['cluster']['lazy_decode']['window']['cluster_reads_per_token']
    else:v=d['arms']['cluster']['curve'][-1]['ce']-d['arms']['baseline']['curve'][-1]['ce']
    same(r['label'],r['y'],[v])
for r in records('lazy_planned_vs_lru'):
    d=load(r['source']);x=[];y=[]
    for row in d['rows']:
        for key,acc in [('plain',x),('planned',y)]:
            acc.append(row[key]['s'] if r['label']=='s' else row[key]['ssd']/row[key]['n_tokens'])
    same(r['label']+' LRU',r['x'],x);same(r['label']+' planned',r['y'],y)
for r in records('speculative_bench'):
    mode,metric=r['label'].split();rows=load(r['source'])['rows']
    v=[a[mode]['n_tokens']/a[mode]['s'] if metric=='throughput' else a[mode]['verify_ready_rate'] for a in rows]
    same(r['label'],r['y'],v)
for r in records('long_context_memory'):
    d=load(r['source']);same(r['label']+' reported table',r['y'],d[r['json_fields'][0]])
for r in records('long_context_throughput'):
    d=load(r['source']);field=r['json_fields'][0].split('.')[-1]
    label=r['label'][:-len(field)].strip()
    arm={'Baseline':'base','Long-context':'long','+ block recompute':'long+recompute_block','+ MLP recompute':'long+recompute_mlp'}[label]
    expected=[]
    for s in r['x']:
        row=next((a for a in d['real_corpus'] if a['arm']==arm and a['seq']==s),None)
        expected.append(None if row is None else row[field]/(1000 if field=='tok_per_s' else 1))
    same(r['label'],r['y'],expected)

manifest=json.loads((out/'figure_manifest.json').read_text())
assert manifest['rendered']>=18
assert all(not r['layout']['issues'] for r in manifest['figures'] if r['status']=='rendered')
report={'status':'passed','numeric_checks':len(checks),'checked_numeric_entries':point_count,
        'method':'Independent comparison of exported plotted data with original JSONs; dt tolerance 1e-12, other numeric checks exact.',
        'checks':checks,'source_measurements_rerun':False,
        'reported_data_boundary':'Single-layer memory values retain reported/unverified status; no raw SWA measurement record was supplied.'}
(package/'data_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','numeric_checks','checked_numeric_entries']}))
