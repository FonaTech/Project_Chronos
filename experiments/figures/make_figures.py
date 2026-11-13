#!/usr/bin/env python3
"""Readable, traceable Chronos figures at 180 mm publication width.

Run `python3 make_figures.py --repo /path/to/Chronos` or use the bundled
source_data snapshot. No model execution or network access is required.
Raw samples are preserved: no smoothing, invented replicates or error bars.
Missing optional benchmarks are recorded, never filled with dummy values.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"chronos-figure-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter, FuncFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE/"source_data"
OUT=HERE/"figures"
DPI=600
FORMATS=("png","pdf","svg")
REGISTRY={}
CURRENT={}
RESULTS=[]
SOURCE_FILES={}
MM=1/25.4
INK="#253243"; MUTED="#617080"; RULE="#D9E0E6"
OKABE={"blue":"#0072B2","orange":"#D68E00","green":"#009E73","sky":"#56B4E9",
       "vermil":"#D55E00","purple":"#9C5A91","yellow":"#C8B530","black":INK,"grey":"#78828C"}
PALETTE=[OKABE[k] for k in ("blue","vermil","green","orange","purple","sky","grey")]
COLOR={"d1":OKABE['grey'],"d4":OKABE['orange'],"d8":OKABE['vermil'],
       "b4":OKABE['blue'],"b8":OKABE['purple'],"b4sig":OKABE['green'],"d4sig":OKABE['sky']}
LABEL={"d1":"R1 · no loop","d4":"R4 · layer","d8":"R8 · layer","b4":"R4 · block",
       "b8":"R8 · block","b4sig":"R4 · block / sigmoid","d4sig":"R4 · layer / sigmoid"}
MARKER={"d1":"o","d4":"s","d8":"^","b4":"D","b8":"v","b4sig":"P","d4sig":"X"}
LINE={"d1":"-","d4":"--","d8":"--","b4":"-","b8":"-","b4sig":"-.","d4sig":"-."}
ARM_PATH={a:f"experiments/loop_depth_15m/{sub}/arm_{a}.json" for a,sub in
          (("d1","results"),("d4","results"),("d8","results"),("b4","results_b"),
           ("b8","results_b"),("b4sig","results_k"),("d4sig","results_m"))}
MODES=["plain","planned","spec-window","spec-planned"]
MODE_LABEL=["LRU","Next-use","Spec.\nwindow","Spec.\nnext-use"]
MODE_COLORS=[OKABE[k] for k in ("grey","blue","orange","purple")]


class MissingData(FileNotFoundError): pass


def figure(name,title,scope,evidence="measured"):
    def register(fn):
        REGISTRY[name]=(fn,title,scope,evidence)
        return fn
    return register


def set_style():
    rcParams.update({"figure.dpi":160,"savefig.dpi":DPI,"font.family":"DejaVu Sans",
        "font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,"axes.linewidth":.65,
        "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":INK,
        "axes.labelcolor":INK,"text.color":INK,"xtick.color":MUTED,"ytick.color":MUTED,
        "xtick.labelsize":7.3,"ytick.labelsize":7.3,"xtick.major.width":.6,"ytick.major.width":.6,
        "xtick.major.size":3,"ytick.major.size":3,"axes.axisbelow":True,
        "legend.frameon":False,"legend.fontsize":7.3,"lines.linewidth":1.35,
        "lines.markersize":3.4,"grid.color":RULE,"grid.linewidth":.5,"grid.alpha":.7,
        "svg.fonttype":"none","pdf.fonttype":42,"mathtext.fontset":"dejavusans",
        "figure.facecolor":"white","savefig.facecolor":"white"})


def load_json(relative):
    path=REPO/relative
    if not path.is_file(): raise MissingData(f"Missing source: {relative}")
    raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    SOURCE_FILES[relative]={"sha256":digest,"bytes":len(raw)}
    CURRENT.setdefault("sources",{})[relative]=digest
    return json.loads(raw)


def load_arm(a): return load_json(ARM_PATH[a])


def record(label,x,y,source,fields,transform="identity",evidence=None,**extra):
    def plain(value):
        if isinstance(value,np.ndarray): return [plain(v) for v in value.tolist()]
        if isinstance(value,(list,tuple,range)): return [plain(v) for v in value]
        if isinstance(value,(np.integer,np.floating)): value=value.item()
        if isinstance(value,float) and not math.isfinite(value): return None
        return value
    CURRENT.setdefault("series",[]).append(dict(label=label,x=plain(x),y=plain(y),source=source,
        json_fields=fields,transform=transform,evidence=evidence or CURRENT['evidence'],**extra))


def canvas(title,height=88,rows=1,cols=1,legend=False,width=180):
    set_style()
    fig=plt.figure(figsize=(width*MM,height*MM),dpi=160)
    fig.text(.07,.956,title,fontsize=11,weight="bold",va="top",ha="left")
    fig.text(.07,.905,CURRENT['scope'],fontsize=6.8,color=MUTED,va="top")
    top=.82 if rows==1 else .84
    bottom=.24 if legend else .19
    gs=fig.add_gridspec(rows,cols,left=.10,right=.972,bottom=bottom,top=top,
                         hspace=.65 if rows>1 else .3,wspace=.42 if cols>1 else .25)
    axes=np.asarray([[fig.add_subplot(gs[r,c]) for c in range(cols)] for r in range(rows)])
    evidence=CURRENT['evidence'].upper().replace('-',' ')
    fig.text(.07,.027,evidence+"  |  "+CURRENT['note'],fontsize=6.4,color=MUTED,va="bottom")
    return fig,axes


def panel(ax,letter,title):
    ax.set_title(letter+"  "+title,loc="left",fontweight="bold",pad=8)
    ax.grid(axis="y")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    ax.tick_params(pad=3)


def legend_below(fig,ax,ncol=3,y=.125):
    handles,labels=ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.54,y),
                   ncol=ncol,columnspacing=1.8,handlelength=2.6,handletextpad=.6)


def arm_line(ax,a,x,y,**kwargs):
    return ax.plot(x,y,color=COLOR[a],linestyle=LINE[a],marker=MARKER[a],
                   markerfacecolor=COLOR[a],markeredgecolor='white',markeredgewidth=.5,label=LABEL[a],**kwargs)


def limits(ax,values,pad=.12,zero=False):
    vals=np.asarray([v for v in np.asarray(values,dtype=object).flatten() if v is not None],dtype=float)
    vals=vals[np.isfinite(vals)]
    if not len(vals): return
    lo,hi=float(vals.min()),float(vals.max()); span=max(abs(hi) if zero else hi-lo,abs(hi)*.05,.015)
    ax.set_ylim(0 if zero else lo-pad*span,hi+pad*span)


def value_bars(ax,x,values,colors,fmt=".2f",labels=None):
    bars=ax.bar(x,values,width=.60,color=colors,edgecolor="white",linewidth=.7,zorder=3)
    ax.bar_label(bars,labels=[format(v,fmt) for v in values],padding=4,fontsize=7.5,color=INK)
    limits(ax,values,pad=.22,zero=True)
    ax.set_xticks(x)
    if labels is not None: ax.set_xticklabels(labels)
    return bars


def _audit(fig):
    fig.canvas.draw(); rend=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
    issues=[]; sizes=[]
    hidden_ticks=set()
    for ax in fig.axes:
        for axis,bounds in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            lo,hi=sorted(bounds)
            visible=[]
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-9<=tick.get_loc()<=hi+1e-9:
                    hidden_ticks.update([id(tick.label1),id(tick.label2)])
                elif tick.label1.get_visible() and tick.label1.get_text():
                    visible.append(tick.label1)
            for a,b in zip(visible,visible[1:]):
                if a.get_window_extent(rend).overlaps(b.get_window_extent(rend)):
                    issues.append({'kind':'tick_label_overlap','labels':[a.get_text(),b.get_text()]})
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in hidden_ticks: continue
        bbox=t.get_window_extent(renderer=rend)
        pts=inv.transform([[bbox.x0,bbox.y0],[bbox.x1,bbox.y1]])
        # Ignore unused or intentionally suppressed tick artists.
        if bbox.width<=0 or bbox.height<=0: continue
        sizes.append(t.get_fontsize())
        if pts[:,0].min()<-.003 or pts[:,0].max()>1.003 or pts[:,1].min()<-.003 or pts[:,1].max()>1.003:
            issues.append({"kind":"page_text_overflow","text":t.get_text()})
    # Check title and figure legend extents against axes plotting areas.
    for leg in fig.legends:
        lb=leg.get_window_extent(rend)
        for ax in fig.axes:
            if lb.overlaps(ax.get_window_extent(rend)):
                issues.append({"kind":"legend_over_plot"})
    return {"min_font_pt":min(sizes) if sizes else None,"issues":issues,
            "size_mm":[v/MM for v in fig.get_size_inches()]}


def publish(fig):
    name=CURRENT['name']; OUT.mkdir(parents=True,exist_ok=True)
    audit=_audit(fig)
    if audit['issues']:
        (OUT/(name+'_layout_error.json')).write_text(json.dumps(audit,indent=2))
        plt.close(fig)
        raise ValueError(f"Layout audit failed for {name}: {audit['issues']}")
    for ext in FORMATS:
        fig.savefig(OUT/(name+'.'+ext),dpi=DPI,bbox_inches=None,pad_inches=0)
        if ext=='svg':
            with plt.rc_context({'svg.fonttype':'path'}):
                fig.savefig(OUT/(name+'_outlined.svg'),bbox_inches=None,pad_inches=0)
    plt.close(fig)
    meta={**CURRENT,'layout':audit,'status':'rendered'}
    (OUT/'data').mkdir(exist_ok=True)
    (OUT/'data'/(name+'.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    RESULTS.append(meta)
    print('Rendered:',name)


def _res_bench():
    candidates=['reports/residency_20261107/bench_residency_50m_s13996_flores16_cpu.json',
                'reports/residency_20261107/bench_residency_50m_s13996_cpu.json']
    for name in candidates:
        if (REPO/name).exists():
            CURRENT['bench_source']=name
            return load_json(name)
    raise MissingData('Missing residency benchmark JSON')


@figure('loop_depth_efficiency','Training cost across recurrent depths',
        '15M backbone · context 512 · one seed per arm · ≈30 min training')
def fig_loop_depth_efficiency():
    arms=['d1','d4','d8']; data=[load_arm(a) for a in arms]
    values=[[d['final']['step']/(d['final']['train_wall_s']/60) for d in data],
            [d['final']['peak_gb'] for d in data],[d['final']['tokens']/1e6 for d in data]]
    fig,axes=canvas(CURRENT['title'],height=85,cols=3)
    for i,(ax,ys,ylab,title) in enumerate(zip(axes.flat,values,
            ['Optimizer steps / min','Peak training memory (reported GB)','Training tokens (million)'],
            ['Throughput','Memory','Tokens processed'])):
        panel(ax,chr(97+i),title)
        value_bars(ax,np.arange(3),ys,[COLOR[a] for a in arms],fmt='.1f' if i!=1 else '.2f',
                   labels=['R = 1','R = 4','R = 8'])
        ax.set_ylabel(ylab,fontsize=7.2 if i==1 else 8)
        for a,v in zip(arms,ys): record(f'{a}: {title}',[a],[v],ARM_PATH[a],
            ['final.step','final.train_wall_s','final.peak_gb','final.tokens'],
            transform=['step / (train_wall_s / 60)','peak_gb','tokens / 1e6'][i])
    publish(fig)


@figure('loop_depth_ce_curves','Validation cross-entropy across loop configurations',
        '15M backbone · one seed per arm · full trajectories and shared training windows')
def fig_loop_depth_ce_curves():
    arms=['d1','d4','d8','b4','b8']; data={}
    for a in arms:
        d=load_arm(a); data[a]={'step':np.array([e['step'] for e in d['evals']]),
             'wall':np.array([e['train_wall_s']/60 for e in d['evals']]),
             'ce':np.array([e['valid_ce'] for e in d['evals']])}
        record(a,data[a]['step'],data[a]['ce'],ARM_PATH[a],['evals[*].step','evals[*].valid_ce'],
               train_minutes=data[a]['wall'].tolist())
    fig,axes=canvas(CURRENT['title'],height=151,rows=2,cols=2,legend=True)
    zoom_hi=min(800,min(d['step'].max() for d in data.values()))
    zoom_lo=min(400,max(0,zoom_hi-300))
    for idx,ax in enumerate(axes.flat):
        field='step' if idx%2==0 else 'wall'; zoom=idx>=2; ys_all=[]
        for a in arms:
            d=data[a]; mask=np.ones(len(d['ce']),dtype=bool)
            if zoom: mask=(d[field]>=zoom_lo)&(d[field]<=zoom_hi) if field=='step' else (d[field]>=20)
            arm_line(ax,a,d[field][mask],d['ce'][mask]); ys_all.extend(d['ce'][mask])
        panel(ax,chr(97+idx),['By optimizer step','By elapsed training time',
              f'Shared-step zoom: {zoom_lo:g}–{zoom_hi:g}','Late-time zoom: 20 min onward'][idx])
        ax.set_ylabel('Validation CE ↓');ax.set_xlabel('Optimizer step' if field=='step' else 'Training time (min)')
        if zoom:
            limits(ax,ys_all)
            ax.set_xlim(zoom_lo,zoom_hi) if field=='step' else ax.set_xlim(20,max(d['wall'].max() for d in data.values()))
        ax.xaxis.set_major_locator(MaxNLocator(4,steps=[1,2,5,10]))
    legend_below(fig,axes[0,0],ncol=3,y=.075)
    publish(fig)


@figure('per_source_ce','Final validation CE by corpus',
        '≈30 min per arm · one seed · row labels include completed optimizer steps')
def fig_per_source_ce():
    arms=['d1','d4','b4','d8','b8'];sources=['fw_edu','edu_zh','cosmo','wiki_en','wiki_zh']
    data=[load_arm(a) for a in arms]
    mat=np.array([[d['final']['per_source_ce'][s] for s in sources] for d in data])
    fig=plt.figure(figsize=(180*MM,104*MM),dpi=160)
    fig.text(.05,.955,CURRENT['title'],fontsize=11,weight='bold',va='top')
    fig.text(.05,.885,CURRENT['scope'],fontsize=6.8,color=MUTED,va='top')
    ax=fig.add_axes([.25,.17,.68,.62])
    cmap=LinearSegmentedColormap.from_list('ce',['#F4F8FC','#BCD5E8','#3979AA','#143E63'])
    # Mesh cells stay vector objects in PDF/SVG; no raster heatmap insert.
    edges=np.arange(-.5,5.5,1)
    ax.pcolormesh(edges,edges,mat,cmap=cmap,vmin=mat.min(),vmax=mat.max(),shading='flat')
    ax.set_xlim(-.5,4.5);ax.set_ylim(4.5,-.5)
    ax.set_xticks(range(5),['FineWeb-Edu','Edu-ZH','Cosmopedia','Wiki EN','Wiki ZH'])
    ax.set_yticks(range(5),[LABEL[a]+f"\n{d['final']['step']:,} steps" for a,d in zip(arms,data)])
    ax.tick_params(axis='both',length=0,pad=7,labelsize=7.6)
    for i,j in np.ndindex(mat.shape):
        ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=9,
                color='white' if (mat[i,j]-mat.min())/(np.ptp(mat)+1e-12)>.58 else INK)
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    fig.text(.25,.085,'CE shown to three decimals · lower is better · darker cells indicate higher CE',fontsize=6.8,color=MUTED)
    fig.text(.05,.027,'MEASURED  |  One run per configuration; no uncertainty estimate.',fontsize=6.4,color=MUTED)
    for a,row in zip(arms,mat):record(a,sources,row,ARM_PATH[a],['final.per_source_ce','final.step'])
    publish(fig)


@figure('per_source_ce_eq_step','Validation CE at matched optimizer steps',
        '15M backbone · nearest observed evaluation to step 800 · lower is better')
def fig_per_source_ce_eq_step():
    arms=['d1','d4','b4','d8','b8']; picks=[]
    for a in arms:
        d=load_arm(a);pick=min(d['evals'],key=lambda e:abs(e['step']-800));picks.append(pick)
        record(a,[pick['step']],[pick['valid_ce']],ARM_PATH[a],['evals[*].step','evals[*].valid_ce'],'nearest observed evaluation to optimizer step 800')
    fig,axes=canvas(CURRENT['title'],height=89);ax=axes[0,0]
    y=np.arange(len(arms))
    for i,(a,p) in enumerate(zip(arms,picks)):
        ax.scatter(p['valid_ce'],i,color=COLOR[a],s=46,marker=MARKER[a],edgecolor='white',linewidth=.6,zorder=4)
        ax.annotate(f"{p['valid_ce']:.3f}  ·  step {p['step']}",(p['valid_ce'],i),xytext=(8,0),
                    textcoords='offset points',va='center',fontsize=7.2)
    vals=[p['valid_ce'] for p in picks];span=max(vals)-min(vals)
    ax.set_xlim(min(vals)-.12*span,max(vals)+.73*span)
    ax.set_yticks(y,[LABEL[a] for a in arms]);ax.invert_yaxis()
    ax.set_xlabel('Validation CE ↓');ax.grid(axis='x');ax.set_axisbelow(True)
    # Dots, not zero-baseline bars, support this explicitly zoomed CE scale.
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_position([.25,.23,.70,.55])
    publish(fig)


@figure('block_vs_layer_ce','Block-order and layer-order training',
        'Depth 4 · clamp and sigmoid step sizes · paired gap uses exact common optimizer steps')
def fig_block_vs_layer_ce():
    arms=['d1','d4','b4','d4sig','b4sig'];fig,axes=canvas(CURRENT['title'],height=104,cols=2,legend=True)
    da={a:load_arm(a) for a in arms}
    for a in arms:
        xs=[e['step'] for e in da[a]['evals']];ys=[e['valid_ce'] for e in da[a]['evals']]
        arm_line(axes[0,0],a,xs,ys);record(a,xs,ys,ARM_PATH[a],['evals[*].step','evals[*].valid_ce'])
    ax=axes[0,0];panel(ax,'a','Observed trajectories');ax.set_xlabel('Optimizer step');ax.set_ylabel('Validation CE ↓')
    aa={e['step']:e['valid_ce'] for e in da['d4sig']['evals']};bb={e['step']:e['valid_ce'] for e in da['b4sig']['evals']}
    steps=sorted(set(aa)&set(bb));gap=[aa[s]-bb[s] for s in steps]
    ax=axes[0,1];panel(ax,'b','Sigmoid: paired CE gap')
    ax.axhline(0,color=MUTED,lw=.8);ax.plot(steps,gap,color=OKABE['blue'],marker='o')
    ax.set_xlabel('Optimizer step');ax.set_ylabel('CE(layer) − CE(block)')
    record('paired sigmoid gap',steps,gap,[ARM_PATH['d4sig'],ARM_PATH['b4sig']],['evals[*].step','evals[*].valid_ce'],
           'exact shared-step intersection; layer CE minus block CE')
    legend_below(fig,axes[0,0],ncol=3,y=.065);publish(fig)


@figure('dt_learned','Learned recurrent step sizes',
        'Sigmoid parameterization · all physical recurrent groups · full curves and late-step detail')
def fig_dt_learned():
    fig,axes=canvas(CURRENT['title'],height=143,rows=2,cols=2,legend=True)
    for col,a in enumerate(['b4sig','d4sig']):
        d=load_arm(a);cfg=d['config'];lo=cfg['recurrent_dt_min'];hi=cfg['recurrent_dt_max']
        n=max(len(e.get('dt',[])) for e in d['evals'])
        for layer in range(n):
            xs=[];ys=[]
            for e in d['evals']:
                raw=e.get('dt',[])
                if len(raw)<=layer or raw[layer] is None:continue
                v=raw[layer];sig=1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
                xs.append(e['step']);ys.append(lo+(hi-lo)*sig)
            for row,ax in enumerate(axes[:,col]):
                use=np.ones(len(xs),dtype=bool) if row==0 else (np.array(xs)>=500)&(np.array(xs)<=1000)
                ax.plot(np.array(xs)[use],np.array(ys)[use],color=PALETTE[layer%len(PALETTE)],
                        marker=['o','s','^','D'][layer%4],mec='white',mew=.5,label=f'Group {layer}')
            record(f'{a}, group {layer}',xs,ys,ARM_PATH[a],['evals[*].dt','evals[*].step','config.recurrent_dt_min','config.recurrent_dt_max'],
                   'dt_min + (dt_max - dt_min) * sigmoid(raw dt)')
        for row,ax in enumerate(axes[:,col]):
            panel(ax,chr(97+row*2+col),('Block order' if col==0 else 'Layer order')+('' if row==0 else ' · steps 500–1000'))
            ax.set_ylabel(r'Effective $\Delta t_{\ell}$');ax.set_xlabel('Optimizer step')
            if row:ax.set_xlim(500,1000)
    legend_below(fig,axes[0,0],ncol=4,y=.075);publish(fig)


@figure('residency_bench','Expert residency policies on a fixed checkpoint',
        '50M checkpoint · FLORES+ en→zh · 16 cold-start prompts · CPU / float32')
def fig_residency_bench():
    d=_res_bench();source=CURRENT['bench_source'];fig,axes=canvas(CURRENT['title'],height=94,cols=3)
    for i,(metric,label,title) in enumerate([
            ('ssd_loads_per_token','Synchronous SSD loads / token ↓','SSD demand loads'),
            ('tok_s','Tokens / second ↑','Throughput'),
            ('selection_miss_rate','Selection miss rate ↓','Routing misses')]):
        vals=[d['summary'][m][metric] for m in MODES];ax=axes[0,i]
        panel(ax,chr(97+i),title)
        value_bars(ax,np.arange(4),vals,MODE_COLORS,fmt='.2f' if i!=2 else '.1%',labels=MODE_LABEL)
        ax.tick_params(axis='x',labelsize=6.6);ax.set_ylabel(label,fontsize=7.2)
        if i==2:ax.yaxis.set_major_formatter(PercentFormatter(1))
        record(metric,MODES,vals,source,[f'summary.*.{metric}'],scope='Summary totals; SSD loads are demand-miss counters, not cluster-file read counts')
    publish(fig)


@figure('cluster_locality_sweep','Cluster-locality coefficient sweep',
        'Synthetic stream benchmark · one seed · routing windows match 32-token topic segments')
def fig_residency_sweep():
    files=sorted((REPO/'reports/residency_20261107/sweep').glob('lam_*_w32.json'))
    if not files:raise MissingData('No lambda sweep JSONs')
    rows=[]
    for path in files:
        rel=str(path.relative_to(REPO));d=load_json(rel);lam=d['settings']['lam']
        # Preserve the original comparison: final CE from each recorded curve.
        cluster=d['arms']['cluster'];base=d['arms']['baseline']
        rows.append((lam,cluster['lazy_decode']['window']['cluster_reads_per_token'],
                     cluster['curve'][-1]['ce']-base['curve'][-1]['ce'],rel))
        record(f'lambda {lam}: reads',[lam],[rows[-1][1]],rel,
               ['settings.lam','arms.cluster.lazy_decode.window.cluster_reads_per_token'],
               'identity: cluster_reads_per_token',scope='Synthetic task; software-instrumented cluster reads')
        record(f'lambda {lam}: CE gap',[lam],[rows[-1][2]],rel,
               ['arms.cluster.curve[-1].ce','arms.baseline.curve[-1].ce'],'final cluster CE - final baseline CE')
    rows.sort();zero=load_json(rows[0][3])
    baseline=zero['arms']['baseline']['lazy_decode']['window']['cluster_reads_per_token']
    x=[0.]+[r[0] for r in rows];reads=[baseline]+[r[1] for r in rows]
    fig,axes=canvas(CURRENT['title'],height=91,cols=2)
    ax=axes[0,0];panel(ax,'a','Logged cluster-page reads')
    ax.plot(range(len(x)),reads,color=OKABE['blue'],marker='o');ax.set_yscale('log')
    ax.set_xlabel(r'Tested $\lambda$ (equally spaced)');ax.set_ylabel('Cluster reads / token (log) ↓')
    ax.set_xticks(range(len(x)),[f'{v:g}' for v in x]);ax.tick_params(axis='x',labelsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax=axes[0,1];panel(ax,'b','Final recorded CE difference')
    ax.axhline(0,color=MUTED,lw=.8);ax.plot(range(len(rows)),[r[2] for r in rows],color=OKABE['vermil'],marker='s')
    ax.set_xticks(range(len(rows)),[f'{r[0]:g}' for r in rows])
    ax.set_xlabel(r'Tested $\lambda$ (equally spaced)');ax.set_ylabel('CE(cluster) − CE(baseline)')
    record('lambda zero baseline',[0],[baseline],rows[0][3],['arms.baseline.lazy_decode.window.cluster_reads_per_token'])
    publish(fig)


@figure('lazy_planned_vs_lru','Paired prompt-level comparison of residency policies',
        '50M checkpoint · 16 paired cold-start prompts · next-use planning versus LRU')
def fig_lazy_vs_full():
    d=_res_bench();source=CURRENT['bench_source'];fig,axes=canvas(CURRENT['title'],height=96,cols=2)
    for i,(field,title,xlab,ylab) in enumerate([
            ('s','Decode latency','LRU time (s)','Next-use time (s)'),
            ('loads','Synchronous demand loads','LRU loads / token','Next-use loads / token')]):
        x=[];y=[]
        for r in d['rows']:
            x.append(r['plain']['s'] if field=='s' else r['plain']['ssd']/r['plain']['n_tokens'])
            y.append(r['planned']['s'] if field=='s' else r['planned']['ssd']/r['planned']['n_tokens'])
        ax=axes[0,i];panel(ax,chr(97+i),title)
        lim=max(x+y)*1.12;ax.plot([0,lim],[0,lim],color=MUTED,ls='--',lw=.8,zorder=1)
        ax.scatter(x,y,s=29,c=OKABE['blue'] if i==0 else OKABE['vermil'],ec='white',lw=.6,zorder=3)
        ax.set_xlim(0,lim);ax.set_ylim(0,lim);ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel(xlab);ax.set_ylabel(ylab)
        ax.text(.04,.96,'Below diagonal: next-use lower',transform=ax.transAxes,va='top',fontsize=6.4,color=MUTED)
        record(field,x,y,source,['rows[*].plain','rows[*].planned'],
               transform='identity latency; per-prompt ssd/n_tokens for loads',paired=True)
    publish(fig)


def box_and_points(ax,groups,labels,colors):
    # Deterministic display offsets; y-values are untouched. Lines connect
    # the same prompt across arms, retaining the paired experimental design.
    offsets=np.linspace(-.10,.10,min(len(g) for g in groups))
    for i,vals in enumerate(zip(*groups)):
        ax.plot(np.arange(1,len(groups)+1)+offsets[i],vals,color=RULE,lw=.65,zorder=1)
    b=ax.boxplot(groups,positions=range(1,len(groups)+1),widths=.42,patch_artist=True,showfliers=False,
        medianprops=dict(color=INK,lw=1.25),boxprops=dict(lw=.8),whiskerprops=dict(lw=.8),capprops=dict(lw=.8))
    for box,color in zip(b['boxes'],colors):box.set(facecolor=color,alpha=.20,edgecolor=color)
    for i,(vals,color) in enumerate(zip(groups,colors),1):
        offsets=np.linspace(-.10,.10,len(vals))
        ax.scatter(i+offsets,vals,s=15,color=color,edgecolor='white',lw=.4,zorder=4)
    ax.set_xticks(range(1,len(groups)+1),labels)


@figure('speculative_bench','Self-speculative decoding and verification readiness',
        '50M checkpoint · FLORES+ en→zh · 16 prompts · every dot is one prompt')
def fig_speculative_bench():
    d=_res_bench();source=CURRENT['bench_source'];rows=d['rows'];fig,axes=canvas(CURRENT['title'],height=96,cols=2)
    modes=['plain','spec-window','spec-planned'];speed=[[r[m]['n_tokens']/r[m]['s'] for r in rows] for m in modes]
    ax=axes[0,0];panel(ax,'a','Per-prompt throughput')
    box_and_points(ax,speed,['LRU','Speculative\nwindow','Speculative\nnext-use'],[MODE_COLORS[i] for i in [0,2,3]])
    ax.set_ylabel('Tokens / second ↑');limits(ax,sum(speed,[]),pad=.18)
    ready=[]
    for mode in modes[1:]:
        vals=[r[mode].get('verify_ready_rate') for r in rows]
        if any(v is None for v in vals):raise MissingData(f'Incomplete verify_ready_rate for {mode}')
        ready.append(vals)
    ax=axes[0,1];panel(ax,'b','Verification-ready fraction')
    box_and_points(ax,ready,['Speculative\nwindow','Speculative\nnext-use'],MODE_COLORS[2:])
    ax.set_ylabel('Verify-ready rate ↑');ax.set_ylim(-.04,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1))
    for mode,vals in zip(modes,speed):record(mode+' throughput',range(len(vals)),vals,source,
        [f'rows[*].{mode}.n_tokens',f'rows[*].{mode}.s'],'n_tokens / seconds',paired=True)
    for mode,vals in zip(modes[1:],ready):record(mode+' readiness',range(len(vals)),vals,source,[f'rows[*].{mode}.verify_ready_rate'],paired=True)
    publish(fig)


@figure('long_context_memory','Reported single-layer attention memory',
        'One attention layer · bf16 · GPU · values reported in source documentation',evidence='reported-unverified')
def fig_long_context_memory():
    rel='long_context_layer_reported.json'
    # Bundled values retain their original provenance; no unsupported OOM bound.
    source=HERE/'source_data'/rel
    if not source.exists():raise MissingData('No cited single-layer memory table')
    raw=source.read_bytes();d=json.loads(raw)
    CURRENT.setdefault('sources',{})['source_data/'+rel]=hashlib.sha256(raw).hexdigest()
    series=[('SWA · full','swa_full_GiB',OKABE['vermil'],'s'),('SWA · blockwise','swa_blockwise_GiB',OKABE['orange'],'s'),
            ('MLA · full','mla_full_GiB',OKABE['blue'],'o'),('MLA · blockwise','mla_blockwise_GiB',OKABE['sky'],'o')]
    fig,axes=canvas(CURRENT['title'],height=96,legend=True);ax=axes[0,0];x=np.array(d['sequence_tokens'])
    for name,key,color,marker in series:
        vals=np.array([np.nan if v is None else v for v in d[key]])
        ax.plot(x,vals,label=name,color=color,marker=marker,mfc='white',ls='--' if 'blockwise' in name else '-')
        for xx,v in zip(x,vals):
            if np.isfinite(v):ax.annotate(f'{v:g}',(xx,v),xytext=(5,6 if 'MLA' in name else -12),
                                         textcoords='offset points',fontsize=6.6,color=color)
        record(name,d['sequence_tokens'],d[key],rel,[key,'sequence_tokens'],scope=d['scope'])
    ax.set_yscale('log');ax.set_xscale('log',base=2);ax.set_ylim(.48,250);ax.set_xlim(14000,105000)
    ax.set_xticks(x,['16,384','32,768','65,536']);ax.set_xlabel('Sequence length (tokens; log₂ spacing)')
    ax.set_ylabel('Extra peak memory (GiB, log)');ax.grid(axis='y')
    ax.text(72000,38,'SWA full:\nreported OOM',fontsize=6.6,color=MUTED,va='center')
    legend_below(fig,ax,ncol=4,y=.095);publish(fig)


@figure('long_context_throughput','Long-context training on real-corpus data',
        '50M random-init model · Apple M5 Max / MLX / bf16 · 3–6 optimizer steps per arm')
def fig_long_context_throughput():
    rel='reports/long_context_20261109/real_corpus_ab.json';d=load_json(rel);rows=d['real_corpus']
    CURRENT['note']='Gaps mean not run, not zero. Brief runs do not establish final model quality.'
    names=[('base','Baseline',OKABE['grey'],'o'),('long','Long-context',OKABE['blue'],'s'),
           ('long+recompute_block','+ block recompute',OKABE['purple'],'^'),
           ('long+recompute_mlp','+ MLP recompute',OKABE['green'],'D')]
    seq=sorted({r['seq'] for r in rows});fig,axes=canvas(CURRENT['title'],height=103,cols=2,legend=True)
    for i,(field,ylabel,title,div) in enumerate([('tok_per_s','Training throughput (k tokens/s) ↑','Throughput',1000),
                                               ('peak_gb','Peak memory (reported GB) ↓','Peak memory',1)]):
        ax=axes[0,i];panel(ax,chr(97+i),title)
        for key,label,color,marker in names:
            vals=[next((r[field]/div for r in rows if r['arm']==key and r['seq']==s),np.nan) for s in seq]
            ax.plot(seq,vals,label=label,color=color,marker=marker,mec='white',mew=.5,ls='--' if 'recompute' in key else '-')
            record(label+' '+field,seq,vals,rel,[f'real_corpus[*].{field}','real_corpus[*].seq','real_corpus[*].arm'],
                   transform=f'{field} / {div}',scope=d['method'])
        ax.set_xscale('log',base=2);ax.set_xticks(seq,['8,192','16,384','32,768','65,536'])
        ax.set_xlabel('Sequence length (tokens; log₂ spacing)');ax.set_ylabel(ylabel,fontsize=7.5)
        ax.set_xlim(min(seq)*.9,max(seq)*1.1);ax.set_ylim(bottom=0)
    legend_below(fig,axes[0,0],ncol=2,y=.055)
    publish(fig)


def _native_publish(draw=None,fn=None):
    import chronos_diagrams as cd
    if fn:
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):
            audit=fn(OUT,formats=FORMATS,dpi=DPI)
        old=audit['figure']
        for ext in FORMATS:
            (OUT/(old+'.'+ext)).replace(OUT/(CURRENT['name']+'.'+ext))
            if ext=='svg':(OUT/(old+'_outlined.svg')).replace(OUT/(CURRENT['name']+'_outlined.svg'))
    else:
        audit=draw.save(OUT,formats=FORMATS,dpi=DPI)
    CURRENT.setdefault('sources',{})['chronos_diagrams.py']=hashlib.sha256((HERE/'chronos_diagrams.py').read_bytes()).hexdigest()
    audit['min_font_pt']=audit['minimum_font_pt']
    meta={**CURRENT,'layout':audit,'status':'rendered'}
    (OUT/'data').mkdir(exist_ok=True)
    (OUT/'data'/(CURRENT['name']+'.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    RESULTS.append(meta)
    print('Rendered:',CURRENT['name'])


@figure('architecture_stack','Configurable forward stack',
        'Source-verified module graph; input lookup and output linear embedding are separate',evidence='schematic')
def fig_architecture_stack():
    import chronos_diagrams as cd
    _native_publish(fn=cd.fig_forward_stack)


@figure('block_vs_layer_schema','Recurrent execution order',
        'Illustrative example: four physical blocks, two repeats; IDs do not denote different layer types',evidence='schematic')
def fig_block_vs_layer_schema():
    import chronos_diagrams as cd
    d=cd.Drawing(180,100,CURRENT['name']);cd._header(d,cd.Architecture(),'Layer order and block order')
    for row,(title,order) in enumerate([('a   LAYER ORDER',[0,0,1,1,2,2,3,3]),('b   BLOCK ORDER',[0,1,2,3,0,1,2,3])]):
        yy=34+row*34;d.text(7,yy-10,title,7.5,weight='bold',ha='left')
        for i,block in enumerate(order):
            x=8+i*15.2
            d.node(x,yy-4,12.2,10,rf'$\ell_{block}$',kind=['attn','local','attn','local'][block],size=9)
            if i<7:d.route([(x+12.2,yy+1),(x+15.2,yy+1)])
        d.text(140,yy-1,'Same parameter block\nreused on each appearance.',6.4,ha='left')
    d.text(7,88,'Example only: G = 4, R = 2. Every physical block still contains attention and an MLP.',6.3,ha='left')
    d.text(7,94,'shared_first: the first iteration writes KV; later iterations recompute queries against the same KV view.',6.3,ha='left',color='muted')
    _native_publish(draw=d)


@figure('router_lookahead','Current dispatch and future expert prediction',
        'Separate data/control paths; future prefetch does not directly contribute to the current output',evidence='schematic')
def fig_router_lookahead():
    import chronos_diagrams as cd
    d=cd.Drawing(180,112,CURRENT['name']);cd._header(d,cd.Architecture(),'Dispatch and lookahead are distinct')
    d.text(5,21,'a   CURRENT MoE DISPATCH',7,weight='bold',ha='left')
    d.node(7,40,26,11,'Hidden states',sub=r'$h_{t,\ell}$',size=6.5,subsize=6)
    d.node(46,29,30,11,'Router',kind='router',sub='FP32 softmax / top-k',size=6.5,subsize=6)
    d.node(94,29,34,11,'Routed experts',kind='expert',sub='Weighted selected outputs',size=6.5,subsize=6)
    d.node(94,51,34,11,'Shared experts',kind='shared',sub='Always-active mean',size=6.5,subsize=6)
    d.add(140,45,sigma=True);d.node(150,40,23,10,'MoE output',kind='ffn',size=6.3)
    d.route([(33,45),(39,45)],arrow=False);d.dot(39,45)
    d.route([(39,45),(39,34.5),(46,34.5)])
    d.route([(39,45),(83,45)],arrow=False);d.dot(83,45)
    d.route([(83,45),(83,34.5),(94,34.5)])
    d.route([(83,45),(83,56.5),(94,56.5)])
    d.route([(76,34.5),(80,34.5),(80,24.5),(111,24.5),(111,29)],color='router_edge',dashed=True)
    d.route([(128,34.5),(140,34.5),(140,43.95)])
    d.route([(128,56.5),(140,56.5),(140,46.05)])
    d.route([(141.05,45),(150,45)])
    d.text(5,74,'b   FUTURE EXPERT DEMAND',7,weight='bold',ha='left')
    for x,w,title,sub in [(7,35,'Post-prelude state','Recurrent path, R > 1'),(52,35,'Lookahead head','Predict expert probabilities'),
                           (98,35,'Future expert IDs','t+1, …, t+K'),(145,28,'Prefetch queue','Weight residency')]:
        d.node(x,81,w,12,title,kind='attn',sub=sub,size=6.2,subsize=6)
    for a,b in [(42,52),(87,98),(133,145)]:d.route([(a,87),(b,87)],color='attn_edge',dashed=True)
    d.text(7,102,'The lookahead head predicts future token expert distributions, not a compulsory next-layer cluster ID.',6.2,ha='left',color='muted')
    d.text(7,108,'At R = 1 the observation point is physical block zero. The prefetch lane carries control, not activations.',6.2,ha='left',color='muted')
    _native_publish(draw=d)


@figure('lazy_tiers','Selective expert paging and runtime policies',
        'Code-generated kernel schematic; includes CUDA and MLX distinctions',evidence='schematic')
def fig_lazy_tiers():
    import chronos_diagrams as cd
    _native_publish(fn=cd.fig_ssd_kernel)


@figure('kernel_evolution','Supported execution configurations',
        'Configuration comparison, not an unverified historical mapping of releases',evidence='schematic')
def fig_kernel_evolution():
    import chronos_diagrams as cd
    d=cd.Drawing(180,113,CURRENT['name']);cd._header(d,cd.Architecture(),'Supported execution configurations')
    cards=[('Single-pass dense',['Input embedding','MLA / SWA attention','Dense SwiGLU FFN','Final norm','Output linear head']),
           ('Single-pass MoE',['Input embedding','MLA / SWA attention','Routed + shared MoE','Final norm','Output linear head']),
           ('Recurrent hybrid MoE',['Input embedding','Prelude blocks','Recurrent blocks ×R','Coda + final norm','Output linear head'])]
    for col,(title,labels) in enumerate(cards):
        x=4+58*col;d.frame(x,21,56,76,fill='panel');d.text(x+28,26,title,7,weight='bold')
        for i,label in enumerate(labels):
            yy=32+i*12;kind=['io','attn','ffn' if col==0 else 'moe','norm','io'][i]
            d.node(x+5,yy,46,8,label,kind=kind,size=6.6)
            if i<4:d.route([(x+28,yy+8),(x+28,yy+12)])
    d.text(6,103,'Input and output remain separate modules. Hybrid attention and MoE are explicit configuration choices.',6.3,ha='left')
    d.text(6,109,'The recurrent path adds conditioning, a contracted update and optional halting; storage is a runtime side path.',6.3,ha='left',color='muted')
    _native_publish(draw=d)


@figure('version_timeline','Release dates recorded in the project changelog',
        'Source: CHANGELOG.md; no undocumented stable-release date is inferred',evidence='reported')
def fig_version_timeline():
    import chronos_diagrams as cd
    path=REPO/'CHANGELOG.md'
    if not path.exists():raise MissingData('Missing CHANGELOG.md')
    raw=path.read_bytes();CURRENT.setdefault('sources',{})['CHANGELOG.md']=hashlib.sha256(raw).hexdigest()
    releases=re.findall(r'^## \[([^\]]+)\]\s*[—–-]\s*(\d{4}-\d{2}-\d{2})',raw.decode(),flags=re.M)[::-1]
    if not releases:raise MissingData('No dated release headers')
    cols=4;rows=math.ceil(len(releases)/cols);height=34+rows*29
    d=cd.Drawing(180,height,CURRENT['name']);cd._header(d,cd.Architecture(),'Release history from CHANGELOG.md')
    for i,(version,date) in enumerate(releases):
        row,col=divmod(i,cols);x=5+43.3*col;y=23+29*row
        d.node(x,y,39.5,19,version,kind=['attn','local','ffn'][min(2,row)],sub=date,size=8,subsize=7)
        if col<cols-1 and i<len(releases)-1:d.route([(x+39.5,y+9.5),(x+43.3,y+9.5)])
    d.text(6,height-6,'Chronological release entries; dates are copied from the supplied changelog, not forecast.',6.3,ha='left',color='muted')
    record('release headers',list(range(len(releases))),releases,'CHANGELOG.md',['dated level-two release headers'],evidence='reported')
    _native_publish(draw=d)


def bench_json(name):return load_json('experiments/figures/benches/out/'+name+'.json')


@figure('moe_balance','Expert load before and after training',
        'Recorded benchmark · identical expert-ID ordering before / after training')
def fig_moe_balance():
    d=bench_json('moe_balance');fig,axes=canvas(CURRENT['title'],height=93,cols=2,legend=True)
    maximum=max(max(d['before']),max(d['after']))
    for i,key in enumerate(['before','after']):
        ax=axes[0,i];vals=d[key];panel(ax,chr(97+i),'Random initialization' if i==0 else 'After training')
        ax.bar(range(len(vals)),vals,color=OKABE['blue'],width=.8,zorder=3)
        ax.axhline(sum(vals)/len(vals),color=OKABE['vermil'],ls='--',lw=.85,label='Uniform reference')
        ax.set_ylim(0,maximum*1.16);ax.set_xlabel('Expert ID');ax.set_ylabel('Fraction of routed tokens')
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
        record(key,list(range(len(vals))),vals,'experiments/figures/benches/out/moe_balance.json',[key])
    legend_below(fig,axes[0,0],ncol=1,y=.10);publish(fig)


@figure('router_entropy','Routing entropy and top-one probability',
        'Recorded training trajectories · no smoothing or extrapolation')
def fig_router_entropy():
    d=bench_json('router_entropy');fig,axes=canvas(CURRENT['title'],height=98,cols=2,legend=True)
    for i,(arm,values) in enumerate(d['arms'].items()):
        color=PALETTE[i%len(PALETTE)]
        for j,key in enumerate(['entropy','top1']):
            axes[0,j].plot(values['step'],values[key],label=arm,color=color)
            record(arm+' '+key,values['step'],values[key],'experiments/figures/benches/out/router_entropy.json',[f'arms.{arm}.step',f'arms.{arm}.{key}'])
    axes[0,0].axhline(d['max_entropy'],color=MUTED,ls='--',lw=.8,label='Maximum entropy')
    for i,(ylabel,title) in enumerate([('Routing entropy (nats)','Entropy'),('Top-one share','Largest routing probability')]):
        panel(axes[0,i],chr(97+i),title);axes[0,i].set_xlabel('Optimizer step');axes[0,i].set_ylabel(ylabel)
    axes[0,1].set_ylim(0,1);axes[0,1].yaxis.set_major_formatter(PercentFormatter(1))
    legend_below(fig,axes[0,0],ncol=3,y=.08);publish(fig)


@figure('lazy_inference_trace','Resident expert memory and cache hit rate',
        'Recorded decoding trace · cumulative hit rates are plotted as recorded')
def fig_lazy_inference_memory():
    d=bench_json('lazy_inference_trace');fig,axes=canvas(CURRENT['title'],height=126,rows=2,legend=True)
    for i,(arm,meta) in enumerate(d['arms'].items()):
        for j,key in enumerate(['resident_gb','cumulative_hit_rate']):
            axes[j,0].plot(meta['token'],meta[key],label=arm,color=PALETTE[i%len(PALETTE)])
            record(arm+' '+key,meta['token'],meta[key],'experiments/figures/benches/out/lazy_inference_trace.json',[f'arms.{arm}.token',f'arms.{arm}.{key}'])
    panel(axes[0,0],'a','Resident expert weights');panel(axes[1,0],'b','Cumulative cache hit rate')
    axes[0,0].set_ylabel('Resident memory (source GB)');axes[1,0].set_ylabel('Cumulative hit rate')
    axes[1,0].set_xlabel('Generated token');axes[1,0].set_ylim(0,1);axes[1,0].yaxis.set_major_formatter(PercentFormatter(1))
    legend_below(fig,axes[0,0],ncol=3,y=.075);publish(fig)


@figure('prefetch_depth','Prefetch coverage across lookahead depths',
        'Recorded benchmark points · missing depth settings are not interpolated')
def fig_prefetch_lookahead_depth():
    d=bench_json('prefetch_depth');fig,axes=canvas(CURRENT['title'],height=91,legend=True);ax=axes[0,0]
    for key,name,color,marker in [('hit_rate','Cache hit rate',OKABE['blue'],'o'),('coverage','Reported cluster coverage',OKABE['vermil'],'s')]:
        ax.plot(d['K'],d[key],label=name,color=color,marker=marker)
        record(name,d['K'],d[key],'experiments/figures/benches/out/prefetch_depth.json',['K',key])
    ax.set_xlabel('Lookahead depth K');ax.set_ylabel('Fraction');ax.set_ylim(0,1);ax.set_xticks(d['K'])
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y')
    legend_below(fig,ax,ncol=2,y=.09);publish(fig)


def main():
    global REPO,OUT,DPI,FORMATS,CURRENT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,help='Chronos repository root; default: bundled source_data or ancestor repo')
    parser.add_argument('--outdir',type=Path,default=HERE/'figures')
    parser.add_argument('--dpi',type=int,default=600)
    parser.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf','svg'])
    parser.add_argument('--figures',nargs='+',choices=list(REGISTRY),help='Render selected figures only')
    parser.add_argument('--strict',action='store_true',help='Return a nonzero exit code if any source dataset is missing')
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    if args.list:
        for name in REGISTRY:print(name)
        return 0
    REPO=args.repo.resolve() if args.repo else HERE/'source_data'
    if not REPO.exists():
        REPO=next((p for p in HERE.parents if (p/'reports').is_dir()),REPO)
    OUT=args.outdir.resolve();DPI=args.dpi;FORMATS=tuple(args.formats)
    if not 72<=DPI<=2400:parser.error('dpi must be in [72,2400]')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in args.figures or list(REGISTRY):
        fn,title,scope,evidence=REGISTRY[name]
        CURRENT=dict(name=name,title=title,scope=scope,evidence=evidence,
                     note='Logged values preserved; no smoothing or inferred error bars.',sources={},series=[])
        try:fn()
        except MissingData as err:
            RESULTS.append({**CURRENT,'status':'missing_data','reason':str(err)})
            print('Missing data:',name,'—',err)
    report=dict(repo=str(REPO),dpi=DPI,formats=list(FORMATS),figures=RESULTS,source_files=SOURCE_FILES,
                rendered=sum(r['status']=='rendered' for r in RESULTS),missing=sum(r['status']=='missing_data' for r in RESULTS))
    (OUT/'figure_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (OUT/'figure_index.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['figure','status','evidence','title','scope'])
        writer.writerows([r[k] for k in ['name','status','evidence','title','scope']] for r in RESULTS)
    print(f"Complete: {report['rendered']} rendered; {report['missing']} missing source datasets.")
    return 1 if args.strict and report['missing'] else 0


if __name__=='__main__':
    sys.exit(main())
