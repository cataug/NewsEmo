#!/usr/bin/env bash
# Creates NewsEmo/FIGURES/build_figures.py and renders 30 publication figures.
# Run this file using 'bash NewsEmo_FIGURES_one_command.sh'.
cd /home/tahiti/NewsEmo || { echo '[STOP] NewsEmo root is unavailable'; false; }
mkdir -p FIGURES logs
if [ -f FIGURES/build_figures.py ]; then
  cp -p FIGURES/build_figures.py "FIGURES/build_figures.py.before_atlas_$(date +%Y%m%d_%H%M%S)"
fi
cat > FIGURES/build_figures.py <<'PY_FIGURE_SCRIPT'
#!/usr/bin/env python3
"""NewsEmo | 30-figure research atlas (existing Phase 2 + extension only).

Reproduces the aesthetic of fractional-lora-unlearning/FIGURES:
large DejaVu Sans typography, cream backgrounds, viridis, black outlines,
soft gradients, white annotation badges with black text, no overlapping labels.
Does NOT train, generate predictions, download weights, or edit existing results.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import re
import textwrap
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'FIGURES'
RES = ROOT / 'results'
EXT = RES / 'phase2_extension'
MAN = ROOT / 'cache' / 'manifests'
PRED = ROOT / 'cache' / 'predictions'
C = plt.get_cmap('viridis')
BG = '#FCF8F1'
AX = '#F1F4EF'
GRID = '#D8DED8'
BLACK = '#15191B'
GREY = '#687271'
PAL = [C(p) for p in (.16,.33,.49,.65,.82,.93)]
SRC = {}
MADE = []
FAILED = []
CURRENT = None

plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':17,
    'axes.titlesize':22, 'axes.labelsize':17,
    'xtick.labelsize':14, 'ytick.labelsize':14,
    'legend.fontsize':13.5, 'figure.titlesize':24,
    'axes.linewidth':.9, 'axes.edgecolor':BLACK,
    'text.color':BLACK, 'axes.labelcolor':BLACK,
    'xtick.color':BLACK, 'ytick.color':BLACK,
    'lines.linewidth':2.3, 'pdf.fonttype':42, 'ps.fonttype':42,
    'savefig.pad_inches':.25
})

def read(name,where='results',required=()):
    folder = {'results':RES,'extension':EXT,'manifests':MAN}[where]
    path=folder/name
    if not path.is_file():
        raise FileNotFoundError(f'{path}: required for {CURRENT}')
    a=pd.read_csv(path,keep_default_na=False,low_memory=False)
    miss=set(required)-set(a.columns)
    if miss: raise ValueError(f'{path}: missing columns {sorted(miss)}')
    SRC.setdefault(CURRENT,[]).append(str(path.relative_to(ROOT)))
    return a

def numeric(a,col):
    return pd.to_numeric(a[col],errors='coerce')

def panel(ax,title='',grid='x'):
    ax.set_facecolor(AX)
    for spine in ax.spines.values():
        spine.set_color(BLACK);spine.set_linewidth(.9)
    ax.set_axisbelow(True)
    if grid!='none':ax.grid(axis=grid,color=GRID,linewidth=.85,alpha=.88,zorder=0)
    if title:ax.set_title(title,fontweight='bold',pad=16)
    ax.tick_params(length=4,width=.75)
    return ax

def caption(fig,txt):
    fig.text(.5,.016,txt,ha='center',va='bottom',fontsize=12.5,color=BLACK)

def badge(ax,x,y,s,offset=(0,0),size=12,ha='center'):
    return ax.annotate(str(s),(x,y),xytext=offset,textcoords='offset points',
        ha=ha,va='center',size=size,weight='bold',color=BLACK,
        bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor=BLACK,lw=.65,alpha=.97),
        annotation_clip=False,zorder=10)

def make(n,slug,fig,note):
    stem=f'{n:02d}_{slug}'
    fig.patch.set_facecolor(BG)
    if os.getenv('NEWSEMO_FIGURE_QUICKCHECK'):
        fig.savefig(OUT/(stem+'.png'),dpi=70,facecolor=BG,bbox_inches='tight')
    else:
        fig.savefig(OUT/(stem+'.pdf'),facecolor=BG,bbox_inches='tight')
        fig.savefig(OUT/(stem+'.png'),dpi=250,facecolor=BG,bbox_inches='tight')
    plt.close(fig)
    MADE.append(dict(id=n,stem=stem,note=note,source_files=SRC.get(CURRENT,[])))
    print(f'[SAVED] {stem} PDF + PNG',flush=True)

def compact(name,maxlen=28):
    s=str(name).replace('baseline_','').replace('ablation_','').replace('vlm_','')
    s=re.sub(r'_s\d+$','',s)
    s=re.sub(r'_\d+$','',s)
    s=s.replace('_',' ')
    if len(s)>maxlen:return s[:maxlen-1]+'…'
    return s

def meaningful(a,metric='macro_f1'):
    if 'status' in a:a=a[a.status.astype(str).eq('ok')]
    if metric in a:a=a.assign(**{metric:numeric(a,metric)}).dropna(subset=[metric])
    return a.copy()

def gradient_barh(ax,y,width,height=.55,low=.16,high=.84,alpha=.83,left=0.):
    if width<=0:return
    grad=np.linspace(low,high,128).reshape(1,-1)
    ax.imshow(grad,extent=(left,left+width,y-height/2,y+height/2),cmap='viridis',aspect='auto',
              interpolation='bicubic',alpha=alpha,zorder=3)
    ax.add_patch(Rectangle((left,y-height/2),width,height,facecolor='none',edgecolor=BLACK,lw=.75,zorder=4))

def heat(ax,A,xlabels,ylabels,fmt='.2f',title='',vmin=None,vmax=None):
    panel(ax,title,'none')
    values=np.ma.masked_invalid(np.asarray(A,float))
    if values.count()==0:raise ValueError('Heatmap has no finite values')
    im=ax.imshow(values,cmap='viridis',vmin=vmin,vmax=vmax,aspect='auto',alpha=.88)
    ax.set_xticks(range(len(xlabels)),xlabels)
    ax.set_yticks(range(len(ylabels)),ylabels)
    ax.set_xticks(np.arange(-.5,len(xlabels),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(ylabels),1),minor=True)
    ax.grid(which='minor',color=BLACK,lw=.43,alpha=.55)
    ax.tick_params(which='minor',left=False,bottom=False)
    # White BADGE background, dark text: never white font on dark heatmap.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            if np.isfinite(values[i,j]):
                ax.text(j,i,format(float(values[i,j]),fmt),ha='center',va='center',fontsize=11.2,
                    color=BLACK,weight='bold',bbox=dict(boxstyle='round,pad=.16',fc='white',ec='none',alpha=.83))
    return im

def load_prediction(ds, tag, seed=42):
    p=PRED/ds/f'{tag}_s{seed}.npz'
    if not p.exists():raise FileNotFoundError(p)
    SRC.setdefault(CURRENT,[]).append(str(p.relative_to(ROOT)))
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def confusion(ds,tag):
    z=load_prediction(ds,tag)
    yy=z['test_y'].astype(str);cl=z['classes'].astype(str)
    pp=cl[z['test_pred'].argmax(1)]
    m=np.zeros((len(cl),len(cl)),float)
    index={v:i for i,v in enumerate(cl)}
    for y,p in zip(yy,pp):
        if y in index and p in index:m[index[y],index[p]]+=1
    return cl,m

def seaborn_free_box(ax,g,positions,orient='h',colors=None):
    for k,(v,pos) in enumerate(zip(g,positions)):
        v=np.asarray(v,float);v=v[np.isfinite(v)]
        if not len(v):continue
        m=np.median(v);q1,q3=np.quantile(v,[.25,.75]);c=colors[k%len(colors)] if colors else PAL[k%6]
        if orient=='h':
            ax.plot([v.min(),v.max()],[pos,pos],color=c,lw=9,alpha=.3,solid_capstyle='round')
            ax.plot([q1,q3],[pos,pos],color=c,lw=14,alpha=.83,solid_capstyle='round')
            ax.scatter(v,np.full(len(v),pos),s=55,color=c,edgecolor=BLACK,lw=.7,zorder=4)
            ax.scatter([m],[pos],s=100,color='white',edgecolor=BLACK,lw=1.2,zorder=5)

def top_models(ds,n=8):
    a=meaningful(read('baseline.csv',required=('dataset','feature','model','macro_f1','seed')))
    a=a[a.dataset.eq(ds)].copy()
    a['method']=a['feature'].astype(str)+' + '+a['model'].astype(str)
    g=a.groupby('method').macro_f1.agg(['mean','std','count']).sort_values('mean',ascending=False)
    return g.head(n)

# -------- 01–03: dataset architecture and labels --------
def fig01():
    names=['inews','gne','mirage','clips','semeval']
    data=[]
    for name in names:
        df=read(name+'.csv','manifests',('split','label'))
        vals=df.split.value_counts();k=(7 if name=='semeval' else df.label.nunique());data.append((name,len(df),k,vals.to_dict()))
    fig,ax=plt.subplots(figsize=(15.5,8.3));panel(ax,'01  |  Five complementary news-evaluation corpora','x')
    splits=['train','val','test','test1','test2','test3','test4','test5','external']
    mx=max(n for _,n,_,_ in data)
    for i,(name,n,k,d) in enumerate(data[::-1]):
        left=0
        for j,s in enumerate(splits):
            cnt=d.get(s,0)
            if not cnt:continue
            ax.barh(i,cnt,height=.62,left=left,color=C(.16+.72*j/max(1,len(splits)-1)),edgecolor=BLACK,lw=.65,alpha=.86)
            if cnt/max(n,1)>.105: ax.text(left+cnt/2,i,f'{s}\n{cnt:,}',ha='center',va='center',size=11,color=BLACK,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.83,'pad':2})
            left+=cnt
        ax.annotate(f'  {n:,} records | {k} '+('dimensions' if name=='semeval' else 'labels'),xy=(n,i),xytext=(12,0),textcoords='offset points',
                    va='center',ha='left',fontsize=12.5,fontweight='bold')
    ax.set_yticks(range(5),[x[0].upper() for x in data[::-1]]);ax.set_xlim(0,mx*1.43)
    ax.set_xlabel('Dataset records (stacked by official/current evaluation split)')
    caption(fig,'Emotion-label corpora and auxiliary authenticity/alignment corpora are distinct tasks.')
    fig.subplots_adjust(left=.15,right=.98,bottom=.16,top=.86)
    make(1,'dataset_topology',fig,'Record counts, split composition and classes for five datasets')

def fig02():
    fig,axs=plt.subplots(1,2,figsize=(17,9));
    for ax,ds,title in zip(axs,['inews','gne'],['iNews: 9 human emotion labels','GNE: 15 emotion categories']):
        a=read(ds+'.csv','manifests',('label',));g=a.label.value_counts().head(15).iloc[::-1]
        panel(ax,title,'x');y=np.arange(len(g))
        for j,(name,val) in enumerate(g.items()):gradient_barh(ax,y[j],val,height=.64,low=.13,high=.83)
        ax.set_yticks(y,[compact(k,22) for k in g.index]);ax.set_xlim(0,max(g)*1.22)
        ax.set_xlabel('Number of articles');
        for j,v in enumerate(g.values):ax.text(v+max(g)*.025,j,str(v),va='center',fontsize=11.5)
    caption(fig,'Gold-label frequency, not predicted prevalence; category imbalance informs macro-F1.')
    fig.subplots_adjust(left=.12,right=.98,bottom=.12,top=.87,wspace=.55)
    make(2,'human_label_landscape',fig,'Per-class frequency of human labels in iNews and GNE')

def fig03():
    fig,ax=plt.subplots(figsize=(12,8));panel(ax,'03  |  How concentrated are the label distributions?','both')
    for j,ds in enumerate(['inews','gne','mirage']):
        a=read(ds+'.csv','manifests',('label',));cnt=np.sort(a.label.value_counts().to_numpy());cum=np.r_[0,np.cumsum(cnt)/cnt.sum()];x=np.linspace(0,1,len(cum))
        ax.plot(x,cum,lw=3,marker='o',markersize=7,color=PAL[j*2],label=f'{ds.upper()} · {len(cnt)} labels')
        ax.fill_between(x,cum,x,color=PAL[j*2],alpha=.075)
    ax.plot([0,1],[0,1],ls='--',color=BLACK,lw=1.3,label='Perfectly balanced')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Cumulative share of classes (least → most frequent)');ax.set_ylabel('Cumulative share of examples')
    ax.legend(loc='upper left',facecolor='white',edgecolor=BLACK)
    caption(fig,'Lorenz-style descriptive curves: more concavity implies stronger class imbalance.')
    fig.subplots_adjust(left=.13,right=.97,bottom=.13,top=.86)
    make(3,'label_inequality_curves',fig,'Lorenz curves of class-frequency concentration')

# -------- 04–07: human annotation heterogeneity --------
def fig04():
    a=read('inews_post_vote_consensus.csv','extension',('split','n_votes','entropy','majority_fraction'))
    fig,axs=plt.subplots(1,2,figsize=(17,7.5));
    for ax,sp in zip(axs,['train','test']):
        b=a[a.split.eq(sp)];panel(ax,f'iNews {sp} — individual-vote uncertainty','both')
        x=numeric(b,'n_votes');y=numeric(b,'entropy');h=ax.hexbin(x,y,gridsize=(18,18),mincnt=1,cmap='viridis',alpha=.87,linewidths=.22,edgecolors=BLACK)
        ax.set_xlabel('Votes per publication');ax.set_ylabel('Shannon entropy of emotion votes')
        fig.colorbar(h,ax=ax,pad=.02,label='Number of publications').ax.tick_params(labelsize=11)
    caption(fig,'Post-level heterogeneity; high entropy represents conflicting human judgments.')
    fig.subplots_adjust(left=.09,right=.96,bottom=.14,top=.87,wspace=.33)
    make(4,'post_vote_entropy_map',fig,'Per-post human disagreement as a function of vote count')

def fig05():
    a=read('inews_rater_agreement.csv','extension',('split','krippendorff_alpha_nominal','alpha_cluster_ci_low','alpha_cluster_ci_high','pairwise_exact_agreement'))
    fig,ax=plt.subplots(figsize=(12.8,7.7));panel(ax,'05  |  Human agreement with uncertainty','x')
    order=['train','val','test'];a=a.set_index('split').reindex(order)
    y=np.arange(3)[::-1];colors_=PAL[::2]
    for i,(sp,r) in enumerate(a.iterrows()):
        v=float(r.krippendorff_alpha_nominal);lo=float(r.alpha_cluster_ci_low);hi=float(r.alpha_cluster_ci_high)
        ax.plot([lo,hi],[y[i],y[i]],lw=13,color=colors_[i],alpha=.29,solid_capstyle='round')
        ax.plot([lo,hi],[y[i],y[i]],lw=2,color=colors_[i])
        ax.scatter([v],[y[i]],s=290,c=[colors_[i]],edgecolor=BLACK,lw=1.2,zorder=4)
        badge(ax,v,y[i],f'α={v:.3f}',offset=(10,22),size=13,ha='left')
        ax.scatter([float(r.pairwise_exact_agreement)],[y[i]-.18],s=100,marker='D',c=[C(.9)],edgecolor=BLACK,zorder=4)
    ax.set_yticks(y,[s.upper() for s in order]);ax.set_ylim(-.5,2.75)
    ax.set_xlim(min(0,float(a.alpha_cluster_ci_low.min())-.03),max(.55,float(a.alpha_cluster_ci_high.max())+.04))
    ax.set_xlabel('Agreement coefficient / 95% cluster bootstrap CI')
    ax.legend(handles=[Line2D([0],[0],marker='o',color=PAL[2],lw=7,alpha=.75,label='Krippendorff α ± 95% CI'),
        Line2D([0],[0],marker='D',color=C(.9),lw=0,label='Exact pairwise agreement')],loc='lower right',facecolor='white',edgecolor=BLACK)
    caption(fig,'Train/validation/test have different numbers of raters and votes per post.')
    fig.subplots_adjust(left=.12,right=.95,bottom=.15,top=.86)
    make(5,'rater_agreement_intervals',fig,'Nominal alpha intervals and pairwise agreement')

def fig06():
    a=read('inews_raters.csv','extension',('split','annotations','agreement_with_majority'))
    fig,ax=plt.subplots(figsize=(13,8));panel(ax,'06  |  Annotator workload versus majority-vote agreement','both')
    for j,sp in enumerate(['train','val','test']):
        b=a[a.split.eq(sp)];
        if len(b):ax.scatter(numeric(b,'annotations'),numeric(b,'agreement_with_majority'),s=46+np.sqrt(numeric(b,'annotations'))*12,
                c=[PAL[j*2]]*len(b),alpha=.61,edgecolor=BLACK,lw=.48,label=f'{sp}: n={len(b)}')
    ax.axhline(.5,ls='--',color=GREY,lw=1);ax.set_xscale('symlog',linthresh=3)
    ax.set_xlabel('Number of annotations per rater (symmetric-log scale)');ax.set_ylabel('Agreement with within-post majority')
    ax.set_ylim(-.03,1.05);ax.legend(facecolor='white',edgecolor=BLACK)
    caption(fig,'Observed rater agreement is not a calibrated individual skill estimate.')
    fig.subplots_adjust(left=.12,right=.96,bottom=.15,top=.87)
    make(6,'rater_workload_reliability',fig,'Individual rater sample counts vs majority agreement')

def fig07():
    a=read('inews_human_level_model_scores.csv','extension',('model','seed','individual_macro_f1','aggregate_macro_f1','per_annotation_accuracy'))
    b=a.groupby('model').agg(individual=('individual_macro_f1','mean'),aggregate=('aggregate_macro_f1','mean'),accuracy=('per_annotation_accuracy','mean')).reset_index()
    fig,ax=plt.subplots(figsize=(13,9));panel(ax,'07  |  Evaluation target changes model ranking','both')
    sizes=150+1200*(b['accuracy']-b['accuracy'].min())/max(.001,b['accuracy'].max()-b['accuracy'].min())
    ax.plot([0,.4],[0,.4],color=GREY,ls='--',lw=1.4)
    ax.scatter(b['aggregate'],b['individual'],s=sizes,c=np.linspace(.18,.85,len(b)),cmap='viridis',edgecolor=BLACK,lw=.8,alpha=.82)
    top=b.sort_values('individual',ascending=False).head(5)
    offsets=[(10,21),(10,-18),(-10,24),(-20,-24),(12,12)]
    for k,r in enumerate(top.itertuples()):badge(ax,r.aggregate,r.individual,compact(r.model,22),offset=offsets[k],size=11,ha='left' if offsets[k][0]>0 else 'right')
    ax.set_xlabel('Macro-F1 against majority-vote article labels')
    ax.set_ylabel('Macro-F1 against individual human votes')
    caption(fig,'Bubble area is per-annotation accuracy (relative scaling); each point is a model averaged over 5 seeds.')
    fig.subplots_adjust(left=.12,right=.96,bottom=.16,top=.87)
    make(7,'individual_vs_aggregate_emotion',fig,'Individual-level versus aggregated human-label model evaluation')

# -------- 08–13: classification baseline diagnostics --------
def fig08():
    fig,axs=plt.subplots(1,3,figsize=(21,9))
    for j,(ax,ds) in enumerate(zip(axs,['inews','gne','mirage'])):
        g=top_models(ds,n=9).iloc[::-1];panel(ax,ds.upper()+' · macro-F1','x');y=np.arange(len(g))
        for i,(method,row) in enumerate(g.iterrows()):
            gradient_barh(ax,y[i],row['mean'],height=.58,low=.12+.05*j,high=.73+.06*j)
            if np.isfinite(row['std']):ax.errorbar(row['mean'],y[i],xerr=row['std'],color=BLACK,fmt='o',markersize=4,capsize=3,zorder=5)
        ax.set_yticks(y,[compact(v,25) for v in g.index]);ax.set_xlim(0,max(.22,g['mean'].max()*1.22));ax.set_xlabel('Test macro-F1 ± seed SD')
    caption(fig,'Ranked within each distinct task; absolute cross-dataset scores are not directly comparable.')
    fig.subplots_adjust(left=.11,right=.985,bottom=.12,top=.87,wspace=.70)
    make(8,'baseline_three_domain_leaderboards',fig,'Complete top-9 baseline ranking by corpus with seed SD')

def fig09():
    a=read('baseline.csv',required=('dataset','feature','model','val_macro_f1','macro_f1','worst_class_recall'))
    a=meaningful(a)
    a=a[a.dataset.eq('mirage')].copy()
    a['method']=a.feature.astype(str)+' / '+a.model.astype(str)
    dims=['val_macro_f1','macro_f1','worst_class_recall']
    tab=a.groupby('method')[dims].mean().dropna()
    if len(tab)<5:raise ValueError('Insufficient multi-objective baseline metrics for MiRAGe')
    tab=tab.loc[tab.macro_f1.nlargest(10).index]
    ranks=tab.rank(ascending=False,method='average')
    fig,ax=plt.subplots(figsize=(14.5,9));panel(ax,'09  |  Rank reversals across evaluation objectives','none')
    for k,method in enumerate(tab.index):
        xx=np.arange(3); yy=ranks.loc[method].to_numpy(float)
        ax.plot(xx,yy,color=PAL[k%6],alpha=.83,lw=2.5,marker='o',markersize=10,mec=BLACK,mew=.65)
        if k<8:
            ax.annotate(compact(method,29),(2,yy[-1]),xytext=(12,0),textcoords='offset points',
                        ha='left',va='center',fontsize=10.4)
    ax.set_xlim(-.12,2.80);ax.set_ylim(len(tab)+.65,.45)
    ax.set_xticks(range(3),['Validation F1','Test F1','Worst-class recall'])
    ax.set_ylabel('Rank among the 10 highest-test-F1 MiRAGe baselines (1 = best)')
    ax.grid(axis='y',color=GRID)
    caption(fig,'Validation, overall generalization, and worst-class protection rank the same methods differently.')
    fig.subplots_adjust(left=.11,right=.78,bottom=.15,top=.87)
    make(9,'multiobjective_baseline_rank_flow',fig,'Rank changes among validation F1, test F1 and worst-class recall on MiRAGe')

def fig10():
    a=read('baseline.csv',required=('dataset','feature','model','seed','macro_f1'))
    a=meaningful(a);fig,axs=plt.subplots(1,3,figsize=(19,8))
    for ax,ds in zip(axs,['inews','gne','mirage']):
        b=a[a.dataset.eq(ds)].copy();b['m']=b.feature.astype(str)+' + '+b.model.astype(str)
        best=b.groupby('m').macro_f1.mean().nlargest(6).index.tolist()[::-1]
        panel(ax,ds.upper()+' · seeded model stability','x');groups=[b[b.m.eq(m)].macro_f1.to_numpy() for m in best]
        seaborn_free_box(ax,groups,range(len(best)),colors=[PAL[i%6] for i in range(len(best))])
        ax.set_yticks(range(len(best)),[compact(m,21) for m in best]);ax.set_xlabel('Test macro-F1, one dot per seed')
        if groups:ax.set_xlim(max(0,min(map(np.min,groups))-.055),max(map(np.max,groups))+.075)
    caption(fig,'Multiple identical seed points indicate deterministic fits, not independent replications.')
    fig.subplots_adjust(left=.11,right=.985,bottom=.13,top=.86,wspace=.67)
    make(10,'seed_uncertainty_raincloud',fig,'Seed dispersion of top baselines by task')

def confusion_plot(num,ds,tag,title):
    cl,mat=confusion(ds,tag)
    mat=mat/(mat.sum(axis=1,keepdims=True)+1e-12)
    fig,ax=plt.subplots(figsize=(13.8,10.2) if len(cl)>10 else (11.5,9.5))
    labels=[compact(x,17) for x in cl]
    hm=heat(ax,mat,labels,labels,fmt='.2f',title=title,vmin=0,vmax=1)
    ax.set_xlabel('Predicted emotion');ax.set_ylabel('True emotion')
    plt.setp(ax.get_xticklabels(),rotation=53,ha='right',fontsize=10.5)
    plt.setp(ax.get_yticklabels(),fontsize=11)
    fig.colorbar(hm,ax=ax,pad=.018,shrink=.80,label='Within-class prediction proportion')
    caption(fig,'Row-normalized, cached seed-42 held-out predictions. Darker = lower recall/concentration.')
    fig.subplots_adjust(left=.19,right=.94,bottom=.27,top=.89)
    make(num,('inews' if ds=='inews' else 'gne')+'_best_confusion',fig,'Full per-class confusion geometry of the strongest baseline')

def fig11():confusion_plot(11,'inews','baseline_minilm_mlp_6','11  |  iNews — MiniLM + MLP: error topology')
def fig12():confusion_plot(12,'gne','baseline_minilm_mlp_6','12  |  GNE — MiniLM + MLP: error topology')

def fig13():
    a=read('vad.csv',required=('target','feature','mae','seed'))
    a=a.assign(mae=numeric(a,'mae')).dropna(subset=['mae'])
    tab=a.groupby(['feature','target']).mae.mean().unstack();cols=[x for x in ['V','A','D'] if x in tab]
    fig,ax=plt.subplots(figsize=(12,7));im=heat(ax,tab[cols].to_numpy(),cols,[compact(x,20) for x in tab.index],fmt='.3f',title='13  |  iNews affective dimensions — regression MAE')
    fig.colorbar(im,ax=ax,pad=.02,label='Mean absolute error (lower is better)')
    ax.set_xlabel('Human valence / arousal / dominance target');ax.set_ylabel('Feature family')
    caption(fig,'MAE is on the original annotation scale; do not compare axes without inspecting scale.')
    fig.subplots_adjust(left=.17,right=.94,bottom=.14,top=.87)
    make(13,'vad_dimension_error_atlas',fig,'Affective continuous-label regression by target and model feature')

# -------- 14–20: ablations, vision and external validation --------
def fig14():
    a=read('ablation.csv',required=('dataset','feature','macro_f1'));a=meaningful(a)
    tab=a.groupby(['feature','dataset']).macro_f1.mean().unstack()
    baselines={'inews':'full_text','gne':'full_text','mirage':'late_llm'}
    rows=sorted(set(tab.index));cols=['inews','gne','mirage']
    arr=np.full((len(rows),3),np.nan)
    for j,ds in enumerate(cols):
        if ds not in tab:continue
        base=tab.at[baselines[ds],ds] if baselines[ds] in tab.index else np.nan
        for i,f in enumerate(rows):
            if pd.notna(tab.at[f,ds]):arr[i,j]=tab.at[f,ds]-base
    fig,ax=plt.subplots(figsize=(12.5,max(8,len(rows)*.58)))
    vmax=max(.01,np.nanmax(np.abs(arr)));hm=heat(ax,arr,[x.upper() for x in cols],[compact(x,29) for x in rows],fmt='+.3f',title='14  |  Ablation effects relative to task-specific reference',vmin=-vmax,vmax=vmax)
    fig.colorbar(hm,ax=ax,pad=.02,label='Δ test macro-F1 (zero = reference)')
    caption(fig,'Reference: full_text for emotion tasks; late_llm for MiRAGe. Missing combinations are blank.')
    fig.subplots_adjust(left=.25,right=.93,bottom=.12,top=.91)
    make(14,'ablation_delta_atlas',fig,'Feature ablation changes relative to predefined per-task reference')

def fig15():
    a=read('ablation.csv',required=('dataset','feature','macro_f1'));a=meaningful(a)
    x=a[a.dataset.eq('mirage')].groupby('feature').macro_f1.mean()
    needed=['late_llm','no_visual','no_text','no_sem','no_dino','no_clipimage','no_cliptext','shuffled_image']
    g=x.reindex([z for z in needed if z in x]).dropna().sort_values()
    ref=float(x['late_llm']);d=g-ref
    fig,ax=plt.subplots(figsize=(15,8));panel(ax,'15  |  MiRAGeNews — what does each modality contribute?','x')
    for i,(feature,delta) in enumerate(d.items()):
        c=PAL[4] if delta>=0 else PAL[1];ax.plot([0,delta],[i,i],color=c,lw=9,alpha=.25,solid_capstyle='round')
        ax.scatter([delta],[i],s=175,color=c,edgecolor=BLACK,lw=.9,zorder=4)
        badge(ax,delta,i,f'{delta:+.3f}',offset=(18 if delta>=0 else -18,0),size=11,ha='left' if delta>=0 else 'right')
    ax.axvline(0,color=BLACK,ls='--',lw=1.2)
    ax.set_yticks(range(len(d)),[compact(s,30) for s in d.index]);ax.set_xlim(d.min()-.11,max(.03,d.max()+.08))
    ax.set_xlabel('Δ macro-F1 relative to complete late-fusion + LLM model')
    caption(fig,'Controlled ablations on the same 2,500 held-out MiRAGe pairs (seed 42).')
    fig.subplots_adjust(left=.20,right=.97,bottom=.14,top=.88)
    make(15,'modality_contribution_ladder',fig,'Model-difference map of modality removals and shuffled images')

def fig16():
    a=read('vlm.csv',required=('dataset','feature','macro_f1','seed'));a=meaningful(a)
    g=a[a.dataset.eq('mirage')].groupby('feature').macro_f1.agg(['mean','std']).sort_values('mean')
    if len(g)<3:raise ValueError('Incomplete matched VLM experiment')
    fig,ax=plt.subplots(figsize=(13.4,7.6));panel(ax,'16  |  Fair matched-subset VLM comparison','x')
    vals=np.arange(len(g))
    for i,(name,r) in enumerate(g.iterrows()):
        gradient_barh(ax,i,r['mean'],height=.52,low=.15,high=.86)
        ax.errorbar(r['mean'],i,xerr=0 if np.isnan(r['std']) else r['std'],color=BLACK,fmt='o',capsize=4,lw=.8,zorder=5)
        badge(ax,r['mean'],i,f'{r["mean"]:.4f}',offset=(43,0),size=12)
    ax.set_xlim(0, min(1,float(g['mean'].max()+.19)));ax.set_yticks(vals,[compact(k,28) for k in g.index]);ax.set_xlabel('Macro-F1 on the same VLM-matched test partition')
    caption(fig,'Matched evaluation, not comparable with full 2,500-example MiRAGe score without conditioning on the subset.')
    fig.subplots_adjust(left=.17,right=.97,bottom=.15,top=.87)
    make(16,'vlm_matched_comparison',fig,'Controlled comparison of image-text VLM fusion against frozen embeddings')

def fig17():
    a=read('robustness.csv',required=('dataset','perturbation','macro_f1','seed'));a=meaningful(a)
    b=a[(a.dataset=='mirage')&a.perturbation.astype(str).str.startswith('OOD_')].copy();
    b['domain']=b.perturbation.str.replace('OOD_','',regex=False)
    g=b.groupby('domain').macro_f1.agg(['mean','std','min','max']).sort_values('mean')
    if len(g)!=5:raise ValueError('Expected five MiRAGe OOD groups')
    fig,ax=plt.subplots(figsize=(13,8));panel(ax,'17  |  Domain shift across five image generators / sources','x')
    for i,(d,r) in enumerate(g.iterrows()):
        ax.plot([r['min'],r['max']],[i,i],color=PAL[i],lw=13,alpha=.25,solid_capstyle='round')
        ax.errorbar([r['mean']],[i],xerr=[[r['mean']-r['min']],[r['max']-r['mean']]],fmt='o',color=PAL[i],
                    mec=BLACK,ms=11,capsize=5,lw=1.4,zorder=4)
        badge(ax,r['mean'],i,f'{r["mean"]:.3f}',offset=(0,24),size=12)
    ax.set_yticks(range(len(g)),[x.replace('test','Domain ') for x in g.index]);ax.set_xlim(max(0,g['min'].min()-.1),min(1,g['max'].max()+.1))
    ax.set_xlabel('Test macro-F1; min–max across five seeds')
    caption(fig,'Domain identity is an official test split; seed ranges are not uncertainty intervals over domains.')
    fig.subplots_adjust(left=.16,right=.96,bottom=.14,top=.88)
    make(17,'mirage_ood_domain_fan',fig,'Domain shift and seed dispersion in five fixed OOD test slices')

def fig18():
    a=read('robustness.csv',required=('dataset','perturbation','macro_f1'));a=meaningful(a)
    a=a[a.perturbation.isin(['lowercase','whitespace','punctuation','unicode'])]
    tab=a.groupby(['dataset','perturbation']).macro_f1.mean().unstack()
    cols=['lowercase','whitespace','punctuation','unicode'];idx=['inews','gne','mirage'];tab=tab.reindex(index=idx,columns=cols)
    fig,ax=plt.subplots(figsize=(14,6.5));hm=heat(ax,tab.to_numpy(),[c.capitalize() for c in cols],[x.upper() for x in idx],fmt='.4f',title='18  |  Robustness under surface-level text perturbations')
    fig.colorbar(hm,ax=ax,pad=.02,label='Macro-F1 on perturbed text')
    ax.set_xlabel('Text normalization perturbation');ax.set_ylabel('Dataset')
    caption(fig,'Perturbation macro-F1 is absolute; near-identical scores imply invariance of this TF-IDF baseline.')
    fig.subplots_adjust(left=.12,right=.94,bottom=.16,top=.86)
    make(18,'perturbation_invariance_matrix',fig,'Cross-dataset surface perturbation robustness matrix')

def fig19():
    a=read('external.csv',required=('dataset','method','auc','n'));a=a[a.dataset.isin(['NewsCLIPpings','NewsCLIPpings_VLM_matched'])].copy()
    tab=a.pivot_table(index='method',columns='dataset',values='auc',aggfunc='mean')
    if 'NewsCLIPpings' not in tab:raise ValueError('Missing full NewsCLIPpings AUC')
    tab=tab.sort_values('NewsCLIPpings',ascending=True).tail(12)
    fig,ax=plt.subplots(figsize=(15,9));panel(ax,'19  |  NewsCLIPpings zero-shot alignment: all versus matched','x')
    ypos=np.arange(len(tab));full=tab['NewsCLIPpings'];match=tab.get('NewsCLIPpings_VLM_matched',pd.Series(index=tab.index,dtype=float))
    for i,name in enumerate(tab.index):
        f=full.loc[name];m=match.loc[name] if name in match.index else np.nan
        if np.isfinite(m):ax.plot([f,m],[i,i],lw=3,color=GREY,alpha=.5)
        ax.scatter([f],[i],s=145,c=[PAL[1]],edgecolor=BLACK,lw=.8,zorder=4)
        if np.isfinite(m):ax.scatter([m],[i],s=140,c=[PAL[4]],marker='D',edgecolor=BLACK,lw=.8,zorder=4)
    ax.set_yticks(ypos,[compact(n,29) for n in tab.index]);ax.set_xlabel('ROC-AUC (higher means better separation of falsified pairs)')
    ax.legend(handles=[Line2D([0],[0],marker='o',lw=0,color=PAL[1],markersize=10,label='Full, n=6,925'),
                       Line2D([0],[0],marker='D',lw=0,color=PAL[4],markersize=10,label='Matched, n=1,000')],
                    loc='lower right',facecolor='white',edgecolor=BLACK)
    caption(fig,'No fitted decision threshold on this external test set; transductive CSLS variants flagged in source CSV.')
    fig.subplots_adjust(left=.25,right=.97,bottom=.14,top=.87)
    make(19,'newsclippings_zero_shot_dumbbells',fig,'External pair-alignment AUC on full and fixed VLM subset')

def fig20():
    a=read('semeval_zeroshot.csv',required=('model','emotion','spearman','n'))
    a=a.assign(spearman=numeric(a,'spearman'))
    t=a.pivot_table(index='emotion',columns='model',values='spearman',aggfunc='mean')
    if t.empty:raise ValueError('SemEval table empty')
    fig,ax=plt.subplots(figsize=(max(11,t.shape[1]*2.4+4),9))
    names=[compact(x,22) for x in t.columns]
    im=heat(ax,t.to_numpy(),names,[x.capitalize() for x in t.index],fmt='.2f',title='20  |  Independent SemEval human-intensity correlation',vmin=-1,vmax=1)
    fig.colorbar(im,ax=ax,pad=.02,label='Spearman ρ against external annotators')
    plt.setp(ax.get_xticklabels(),rotation=28,ha='right',fontsize=11.5)
    ax.set_xlabel('Frozen generative model and prompt condition');ax.set_ylabel('Emotion / valence')
    caption(fig,'No model was fitted to SemEval; unavailable Qwen3 prompts remain absent rather than imputed.')
    fig.subplots_adjust(left=.16,right=.93,bottom=.25,top=.89)
    make(20,'semeval_emotion_prompt_correlations',fig,'External per-emotion Spearman correlations by model and prompt')

# -------- 21–26: GNE role extraction / annotation reliability --------
def fig21():
    a=read('gne_role_metrics.csv','extension',('split','role','micro_token_precision','micro_token_recall','micro_token_f1'))
    a=a[a.split.eq('test')].copy();fig,ax=plt.subplots(figsize=(12,8));panel(ax,'21  |  GNE role extraction: precision–recall geometry','both')
    for j,r in enumerate(a.sort_values('role').itertuples()):
        area=600+1700*float(r.micro_token_f1)
        ax.scatter([r.micro_token_recall],[r.micro_token_precision],s=area,c=[PAL[j]],edgecolor=BLACK,lw=1.2,alpha=.77,zorder=4)
        badge(ax,r.micro_token_recall,r.micro_token_precision,str(r.role).capitalize(),
              offset=[(26,21),(-29,-24),(35,-22),(-28,23)][j],ha='left' if j in (0,2) else 'right')
    ax.plot([0,1],[0,1],color=GREY,ls='--',lw=1.2)
    ax.set_xlim(0,.9);ax.set_ylim(0,.9);ax.set_xlabel('Token-level recall');ax.set_ylabel('Token-level precision')
    caption(fig,'Bubble area ∝ observed token-level F1; plotted on the held-out 750 GNE headlines.')
    fig.subplots_adjust(left=.12,right=.96,bottom=.14,top=.88)
    make(21,'gne_role_precision_recall_bubbles',fig,'GNE P/R/F1 geometry for each role')

def fig22():
    a=read('gne_role_metrics.csv','extension',('split','role','micro_token_f1'))
    tab=a.pivot(index='role',columns='split',values='micro_token_f1').reindex(index=['cue','cause','target','experiencer'],columns=['train','val','test'])
    fig,ax=plt.subplots(figsize=(12,7));im=heat(ax,tab.to_numpy(),['Train','Validation','Test'],[s.capitalize() for s in tab.index],fmt='.3f',title='22  |  Role extraction transfers across splits',vmin=0,vmax=.65)
    fig.colorbar(im,ax=ax,pad=.02,label='Token micro-F1')
    caption(fig,'Stable patterns across official splits are descriptive; same frozen Qwen role extractor.')
    fig.subplots_adjust(left=.16,right=.93,bottom=.15,top=.87)
    make(22,'gne_role_generalization_heatmap',fig,'GNE micro-F1 across roles and train/validation/test splits')

def fig23():
    a=read('gne_role_metrics.csv','extension',('split','role','micro_token_f1','exact_span_accuracy','macro_positive_gold_f1'))
    a=a[a.split.eq('test')].sort_values('micro_token_f1');fig,ax=plt.subplots(figsize=(14.5,7.5));panel(ax,'23  |  Boundary localization is harder than approximate overlap','x')
    for i,r in enumerate(a.itertuples()):
        l=float(r.exact_span_accuracy);h=float(r.micro_token_f1)
        ax.plot([l,h],[i,i],color=PAL[i],lw=12,alpha=.25,solid_capstyle='round')
        ax.scatter([l],[i],s=150,c=[PAL[0]],marker='s',edgecolor=BLACK,zorder=4)
        ax.scatter([h],[i],s=165,c=[PAL[4]],edgecolor=BLACK,zorder=5)
        badge(ax,h,i,f'Δ {h-l:+.3f}',offset=(55,0),size=12)
    ax.set_yticks(range(len(a)),a.role.str.capitalize());ax.set_xlim(0,max(.8,a.micro_token_f1.max()+.22));ax.set_xlabel('Test-set score')
    ax.legend(handles=[Line2D([0],[0],marker='s',lw=0,color=PAL[0],markersize=10,label='Exact-span accuracy'),
        Line2D([0],[0],marker='o',lw=0,color=PAL[4],markersize=10,label='Token-level micro-F1')],loc='lower right',facecolor='white',edgecolor=BLACK)
    caption(fig,'Gap highlights approximate phrase identification versus exact annotated character boundaries.')
    fig.subplots_adjust(left=.14,right=.96,bottom=.15,top=.87)
    make(23,'gne_role_boundary_precision_dumbbell',fig,'Exact span vs token overlap per emotional role')

def fig24():
    a=read('gne_role_per_example.csv','extension',('role','split','tp','fp','fn'))
    a=a[a.split.eq('test')];roles=['cue','cause','target','experiencer']
    grp=a.groupby('role')[['tp','fp','fn']].sum().reindex(roles)
    if grp.isna().all().all():raise ValueError('No role confusion counts')
    fig,ax=plt.subplots(figsize=(13.5,7.5));panel(ax,'24  |  Role extraction token error budget','x')
    y=np.arange(4)
    totals=grp.sum(axis=1).to_numpy();safe=np.maximum(1,totals)
    left=np.zeros(4)
    for j,c in enumerate(['tp','fp','fn']):
        frac=grp[c].to_numpy()/safe
        ax.barh(y,frac,left=left,height=.56,color=PAL[j*2],edgecolor=BLACK,lw=.7,label={'tp':'Matched tokens','fp':'Extra predicted tokens','fn':'Missed gold tokens'}[c],alpha=.84)
        for i,v in enumerate(frac):
            if v>.08:ax.text(left[i]+v/2,i,f'{v:.0%}',ha='center',va='center',fontsize=12,weight='bold',bbox={'facecolor':'white','alpha':.81,'edgecolor':'none','pad':2})
        left+=frac
    ax.set_yticks(y,[s.capitalize() for s in roles]);ax.set_xlim(0,1);ax.set_xlabel('Fraction of combined TP / FP / FN token counts')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,facecolor='white',edgecolor=BLACK)
    caption(fig,'Counts aggregate eligible test instances only; not a per-document error frequency.')
    fig.subplots_adjust(left=.16,right=.97,bottom=.23,top=.88)
    make(24,'gne_role_token_error_composition',fig,'FP/FN and overlap token accounting in each emotional role')

def fig25():
    a=read('gne_gold_annotation_alignment_audit.csv','extension',('role','gold_present','gold_alignable','align_rate'))
    a=a.sort_values('align_rate');fig,ax=plt.subplots(figsize=(13,7.8));panel(ax,'25  |  Gold-span alignment quality and annotation prevalence','x')
    for i,r in enumerate(a.itertuples()):
        frac=float(r.align_rate);ax.barh(i,frac,height=.52,color=PAL[i],edgecolor=BLACK,lw=.7,alpha=.82)
        ax.scatter([frac],[i],s=150,color=PAL[i],edgecolor=BLACK,zorder=4)
        badge(ax,frac,i,f'{frac:.2%} · {int(r.gold_alignable):,}/{int(r.gold_present):,}',offset=(65,0),size=11)
    ax.set_yticks(range(len(a)),a.role.str.capitalize());ax.set_xlim(0,1.36);ax.set_xlabel('Alignable annotated mentions / gold-positive examples')
    caption(fig,'A small set of unalignable gold spans is excluded from official role-overlap scores.')
    fig.subplots_adjust(left=.14,right=.97,bottom=.14,top=.86)
    make(25,'gne_gold_span_alignment_coverage',fig,'Alignment yield and sample sizes of GNE gold role spans')

def fig26():
    a=read('gne_intensity_counts.csv','extension',('split','intensity','n'))
    a['intensity']=a.intensity.replace({'':'unspecified'})
    tab=a.pivot_table(index='split',columns='intensity',values='n',aggfunc='sum',fill_value=0)
    tab=tab.reindex(['train','val','test']).fillna(0)
    fig,axs=plt.subplots(1,2,figsize=(17,7));
    ax=axs[0];panel(ax,'Intensity prevalence (absolute)','y')
    for i,(sp,r) in enumerate(tab.iterrows()):
        left=0
        for j,(name,v) in enumerate(r.items()):
            ax.barh(i,v,left=left,height=.58,color=PAL[j%6],edgecolor=BLACK,lw=.65,alpha=.84)
            left+=v
    ax.set_yticks(range(3),['Train','Validation','Test']);ax.set_xlabel('Examples')
    ax=axs[1];panel(ax,'Minority intensity counts (zoom)','y')
    if 'medium' in tab:minor=tab.drop(columns='medium')
    else:minor=tab
    x=np.arange(3);tot=np.zeros(3)
    for j,col in enumerate(minor.columns):
        v=minor[col].to_numpy();ax.bar(x,v,bottom=tot,color=PAL[j%6],edgecolor=BLACK,lw=.6,alpha=.88,label=col);tot+=v
    ax.set_xticks(x,['Train','Validation','Test']);ax.set_ylabel('Non-medium cases (absolute count)')
    ax.legend(facecolor='white',edgecolor=BLACK,fontsize=12)
    caption(fig,'Nearly all GNE labels are medium; model MAE can be beaten by a constant prediction.')
    fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.84,wspace=.28)
    make(26,'gne_intensity_collapse_diagnostic',fig,'Absolute label collapse and magnified minority intensity counts')

# -------- 27–30: multiobjective optimization, source effects and inference --------
def fig27():
    a=read('pareto_hypervolume_exact.csv','extension',('dataset','method','seed','exact_original_reference','exact_relaxed_reference'))
    fig,axs=plt.subplots(1,3,figsize=(19,7.6))
    for ax,ds in zip(axs,['inews','gne','mirage']):
        sub=a[a.dataset.eq(ds)].copy();panel(ax,ds.upper(),'y')
        methods=['random','single','nsga2'];x=np.arange(len(methods))
        for j,col in enumerate(['exact_original_reference','exact_relaxed_reference']):
            mean=sub.groupby('method')[col].mean().reindex(methods);std=sub.groupby('method')[col].std().reindex(methods).fillna(0)
            ax.bar(x+(j-.5)*.32,mean,.31,color=PAL[1+j*3],edgecolor=BLACK,lw=.7,alpha=.85,
                    yerr=std,capsize=4,label='Original reference' if j==0 else 'Relaxed reference')
        ax.set_xticks(x,['Random','Single','NSGA-II'],rotation=20);ax.set_ylabel('Exact 4D dominated hypervolume')
        ax.legend(loc='upper left',fontsize=11,facecolor='white',edgecolor=BLACK)
    caption(fig,'Exact 4D hypervolume; each dataset and reference point has a different objective-space scale.')
    fig.subplots_adjust(left=.06,right=.985,bottom=.17,top=.86,wspace=.29)
    make(27,'pareto_reference_sensitivity',fig,'Exact hypervolume by optimizer and reference point for three datasets')

def fig28():
    a=read('pareto.csv',required=('dataset','method','val_macro_f1','seed','active_features'))
    a=meaningful(a);fig,axs=plt.subplots(1,3,figsize=(19,7.5))
    for ax,ds in zip(axs,['inews','gne','mirage']):
        panel(ax,ds.upper()+' · selected Pareto solutions','both');sub=a[a.dataset.eq(ds)]
        for j,m in enumerate(['random','single','nsga2']):
            b=sub[sub.method.eq(m)]
            if b.empty:continue
            # Test F1 is reported separately from selection-time validation F1.
            y=numeric(b,'macro_f1') if 'macro_f1' in b else pd.Series(np.nan,index=b.index)
            ax.scatter(numeric(b,'val_macro_f1'),y,s=100+numeric(b,'active_features')*60,
                color=PAL[j*2],edgecolor=BLACK,lw=.8,alpha=.82,label=m.upper())
        ax.set_xlabel('Validation macro-F1 (optimization)');ax.set_ylabel('Held-out test macro-F1')
        ax.legend(facecolor='white',edgecolor=BLACK,fontsize=11)
    caption(fig,'Each point is a seed × optimizer selection; bubble area represents number of active feature blocks.')
    fig.subplots_adjust(left=.08,right=.985,bottom=.17,top=.86,wspace=.36)
    make(28,'pareto_validation_test_generalization',fig,'Pareto optimization validation/test performance and chosen complexity')

def fig29():
    a=read('source_holdout.csv',required=('dataset','source','feature','macro_f1','seed'))
    a=meaningful(a);fig,axs=plt.subplots(1,2,figsize=(19,8))
    for ax,ds in zip(axs,['inews','gne']):
        b=a[a.dataset.eq(ds)];piv=b.pivot_table(index='source',columns='feature',values='macro_f1',aggfunc='mean')
        if piv.empty:raise ValueError('Missing source holdout data for '+ds)
        hm=heat(ax,piv.to_numpy(),[compact(x,17) for x in piv.columns],[compact(x,19) for x in piv.index],fmt='.3f',title=ds.upper()+' · held-out sources')
        fig.colorbar(hm,ax=ax,pad=.02,shrink=.65,label='Macro-F1')
        ax.set_xlabel('Transfer feature family');ax.set_ylabel('Unseen training-source partition')
    caption(fig,'Each source is held out from training; results aggregate five seeds per source × method.')
    fig.subplots_adjust(left=.13,right=.98,bottom=.17,top=.85,wspace=.47)
    make(29,'leave_source_out_transfer_matrix',fig,'Source-holdout generalization by corpus and feature family')

def fig30():
    a=read('paired_randomization_BH_exploratory.csv','extension',('dataset','proposed','control','delta_macro_f1','ci_low','ci_high','bh_q_all_exploratory'))
    a=a.copy()
    # Mirror duplicates are same empirical contrast, not independent evidence.
    a['pairkey']=a.apply(lambda r:(r['dataset'],tuple(sorted([str(r['proposed']),str(r['control'])]))),axis=1)
    a=a.drop_duplicates('pairkey',keep='first')
    a['q']=numeric(a,'bh_q_all_exploratory');a['delta']=numeric(a,'delta_macro_f1')
    a=a.dropna(subset=['q','delta']).sort_values('q').head(24).iloc[::-1]
    fig,ax=plt.subplots(figsize=(17,12));panel(ax,'30  |  Exploratory paired effects (mirror contrasts removed)','x')
    y=np.arange(len(a))
    for i,r in enumerate(a.itertuples()):
        col=PAL[4] if r.q<.05 else PAL[1]
        ax.plot([r.ci_low,r.ci_high],[i,i],color=col,lw=7,alpha=.23,solid_capstyle='round')
        ax.plot([r.ci_low,r.ci_high],[i,i],color=col,lw=2)
        ax.scatter([r.delta],[i],color=col,s=130,edgecolor=BLACK,lw=.8,zorder=4)
    labels=[f'{r.dataset.upper()}  {compact(r.proposed,26)}\nvs {compact(r.control,24)}' for r in a.itertuples()]
    ax.set_yticks(y,labels,fontsize=11.0);ax.axvline(0,color=BLACK,lw=1.1,ls='--')
    ax.set_xlabel('Δ macro-F1 (proposed − control), cluster bootstrap 95% CI')
    ax.legend(handles=[Line2D([0],[0],marker='o',lw=3,color=PAL[4],label='BH q < 0.05 in original 64-test family'),
        Line2D([0],[0],marker='o',lw=3,color=PAL[1],label='Not significant')],facecolor='white',edgecolor=BLACK,loc='lower right')
    caption(fig,'Post-hoc comparisons. Mirror rows removed for presentation; original BH q values retain their original 64-test denominator.')
    fig.subplots_adjust(left=.37,right=.97,bottom=.13,top=.91)
    make(30,'paired_inference_forest_atlas',fig,'Cluster-bootstrap intervals with exploratory BH significance (display de-duplicated)')

FUNCTIONS = [fig01,fig02,fig03,fig04,fig05,fig06,fig07,fig08,fig09,fig10,
             fig11,fig12,fig13,fig14,fig15,fig16,fig17,fig18,fig19,fig20,
             fig21,fig22,fig23,fig24,fig25,fig26,fig27,fig28,fig29,fig30]

def main():
    global CURRENT,ROOT,OUT,RES,EXT,MAN,PRED
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--only',nargs='*',type=int,help='Optional figures to rebuild')
    args=ap.parse_args()
    ROOT=args.root.expanduser().resolve();OUT=ROOT/'FIGURES';RES=ROOT/'results';EXT=RES/'phase2_extension';MAN=ROOT/'cache'/'manifests';PRED=ROOT/'cache'/'predictions'
    OUT.mkdir(exist_ok=True,parents=True)
    print('NEWSEMO SCIENTIFIC FIGURE ATLAS — STYLE REF: fractional-lora-unlearning/FIGURES',flush=True)
    print('Project:',ROOT,' | Output:',OUT,flush=True)
    wanted=set(args.only) if args.only else set(range(1,31))
    for j,func in enumerate(FUNCTIONS,1):
        if j not in wanted:continue
        CURRENT=j
        try:
            func()
        except Exception as exc:
            plt.close('all')
            FAILED.append(dict(id=j,error=str(exc),traceback=traceback.format_exc()))
            print(f'[ERROR] FIG {j:02d}: {type(exc).__name__}: {exc}',flush=True)
    manifest={'version':'1.0','root':str(ROOT),'reference_style':['FIGURES/build_figures.py','FIGURES/build_figures_extended.py'],
              'figure_count':len(MADE),'figures':MADE,'failed':FAILED,
              'methodology':'Read-only visualizations of existing results; no imputations, no model reruns.'}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(MADE).to_csv(OUT/'figure_index.csv',index=False)
    # Optional lightweight contact sheet; index only, not counted as one of 30 research figures.
    if MADE and not os.getenv('NEWSEMO_FIGURE_QUICKCHECK'):
        n=len(MADE);nr=math.ceil(n/4);fig,axes=plt.subplots(nr,4,figsize=(20,nr*3.1),facecolor=BG)
        axes=np.array(axes).reshape(nr,4)
        for ax in axes.ravel():ax.set_axis_off()
        for ax,r in zip(axes.ravel(),MADE):
            im=plt.imread(OUT/(r['stem']+'.png'));ax.imshow(im)
            ax.set_title(r['stem'].replace('_',' ')[:36],fontsize=10.5,pad=4,weight='bold')
        fig.subplots_adjust(wspace=.08,hspace=.25,left=.01,right=.99,top=.98,bottom=.01)
        fig.savefig(OUT/'00_gallery_contact_sheet.jpg',dpi=120,facecolor=BG,bbox_inches='tight')
        plt.close(fig)
    print(f'\n[SUMMARY] figures={len(MADE)} failed={len(FAILED)}',flush=True)
    if FAILED:
        print('[ERROR DETAILS] FIGURES/figure_manifest.json',flush=True)
        raise SystemExit(2)
    if not args.only and len(MADE)!=30:raise SystemExit('Expected exactly 30 published figures')
    print(f'[DONE] {len(MADE)} data-grounded PDF + PNG figures in {OUT}',flush=True)

if __name__=='__main__':main()

PY_FIGURE_SCRIPT
PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
if [ -x "$PY" ]; then
  if "$PY" -m py_compile FIGURES/build_figures.py; then
    echo '[OK] Syntax checked. Beginning 30 publication figures.'
    set -o pipefail
    if "$PY" -u FIGURES/build_figures.py 2>&1 | tee -a logs/figures_atlas.log; then
      echo '[SUCCESS] 30 PDF + 30 PNG figures in /home/tahiti/NewsEmo/FIGURES'
    else
      echo '[STOP] At least one figure failed; see FIGURES/figure_manifest.json and logs/figures_atlas.log'
    fi
  else
    echo '[STOP] Syntax check failed; original results not modified.'
  fi
else
  echo '[STOP] Existing .venv_a100 Python not found; nothing installed.'
fi
echo '[TERMINAL ACTIVE]'
