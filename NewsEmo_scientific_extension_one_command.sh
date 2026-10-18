#!/usr/bin/env bash
# Self-contained, no installs, no downloads. Run: bash NewsEmo_scientific_extension_one_command.sh
if [ ! -d /home/tahiti/NewsEmo ]; then
  echo "[ERROR] Project /home/tahiti/NewsEmo not found"
else
cd /home/tahiti/NewsEmo
mkdir -p scripts logs results/phase2_extension cache/phase2_extension
cat > scripts/phase2_scientific_extension.py <<'PY_EXTENSION_SOURCE'
#!/usr/bin/env python3
"""NewsEmo Phase 2 scientific extension v1.0.
Read-only towards existing datasets, manifests, pretrained models and Phase 2 results.
New predictions: cache/phase2_extension; new reports: results/phase2_extension.
No installs/downloads; idempotent JSONL role generation and atomic aggregate CSVs.
"""
import os, sys, re, ast, json, math, hashlib, argparse, time, gc
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

ROOT=Path('/home/tahiti/NewsEmo')
OUT=ROOT/'results'/'phase2_extension'
CACHE=ROOT/'cache'/'phase2_extension'
RAW=ROOT/'datasets'
SEEDS=(42,43,44,45,46)
EMO=('anger','disgust','fear','joy','sadness','surprise')
ROLES=('cue','cause','target','experiencer')

def log(*x): print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def save_csv(df,name):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');df.to_csv(tmp,index=False);os.replace(tmp,p);log('SAVED',p,len(df));return p
def save_json(obj,name):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8');os.replace(tmp,p);return p
def load_manifest(ds):
 p=ROOT/'cache'/'manifests'/f'{ds}.csv'
 if not p.exists():raise FileNotFoundError(f'Already prepared manifest missing: {p}')
 d=pd.read_csv(p,dtype=str,keep_default_na=False)
 if not d.uid.is_unique:raise RuntimeError(f'{ds} manifest has duplicate UID; refusing to guess alignment')
 return d

def raw_inews():
 specs=(('train','train.csv'),('val','dev.csv'),('test','test_cold_start_public.csv'))
 arr=[]
 for split,f in specs:
  p=RAW/'inews_public'/f
  if not p.exists():raise FileNotFoundError(p)
  d=pd.read_csv(p,dtype=str,keep_default_na=False,low_memory=False)
  cols={'Post_ID','Annotator_ID','Discrete'}
  if not cols.issubset(d):raise RuntimeError(f'{p}: missing {cols-set(d.columns)}')
  d=d.rename(columns={'Post_ID':'uid','Annotator_ID':'rater','Discrete':'vote'})
  d['split']=split;d['uid']=d.uid.astype(str);d['vote']=d.vote.astype(str).str.strip()
  d=d[(d.uid!='')&(d.vote!='')].copy()
  arr.append(d[['uid','rater','vote','split']])
 return pd.concat(arr,ignore_index=True)

def k_alpha(votes_by_unit):
 # Krippendorff nominal alpha with varying numbers of ratings per unit.
 # Each coincidence matrix for n ratings is normalized by (n-1).
 coinc=Counter();tot=Counter();N=0.;units=0
 for vs in votes_by_unit:
  if len(vs)<2:continue
  c=Counter(vs);n=len(vs);units+=1
  for a,na in c.items():
   for b,nb in c.items():
    coincid=na*(nb-(a==b))/(n-1)
    coinc[(a,b)]+=coincid
    tot[a]+=coincid
  N+=n
 if N<2:return float('nan'),units
 do=sum(v for (a,b),v in coinc.items() if a!=b)/N
 de=1-sum(v*(v-1) for v in tot.values())/(N*(N-1))
 return (1-do/de if de>1e-12 else float('nan')),units

def load_prediction(ds,tag,seed):
 path=ROOT/'cache'/'predictions'/ds/f'{tag}_s{seed}.npz'
 if not path.is_file():return None
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def prediction_files(ds,suite='baseline'):
 return sorted((ROOT/'cache'/'predictions'/ds).glob(f'{suite}_*_s*.npz'))

def conf_mat(true,pred,classes):
 cls={str(v):i for i,v in enumerate(classes)};cm=np.zeros((len(cls),len(cls)),dtype=np.int64)
 for a,b in zip(true,pred):
  if str(a) in cls and str(b) in cls:cm[cls[str(a)],cls[str(b)]]+=1
 return cm

def macro_cm(cm):
 tp=np.diag(cm).astype(float);den=cm.sum(0)+cm.sum(1)
 return float(np.mean(np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0)))

def inews_humans():
 log('START INEWS individual-rater and agreement analysis')
 raw=raw_inews();man=load_manifest('inews')
 merged=raw.merge(man[['uid','split','label']],on=['uid','split'],how='inner',validate='many_to_one')
 if merged.empty:raise RuntimeError('No individual iNews votes matched the existing manifest')
 overlap=merged.groupby('uid').split.nunique()
 if (overlap>1).any():raise RuntimeError('iNews post IDs overlap across splits after join')
 all_rows=[];raters=[]
 for sp,g in merged.groupby('split',sort=False):
  groups=list(g.groupby('uid',sort=False).vote.agg(list));sizes=np.array([len(x) for x in groups])
  alpha,n_multi=k_alpha(groups)
  disagree=np.mean([len(set(x))>1 for x in groups])
  pair_good=sum(sum(z*(z-1) for z in Counter(x).values()) for x in groups)
  pair_all=sum(len(x)*(len(x)-1) for x in groups)
  rng=np.random.default_rng(900+{'train':1,'val':2,'test':3}.get(sp,0));ci=[]
  for _ in range(500):
   sample=rng.choice(len(groups),size=len(groups),replace=True)
   estimate,_=k_alpha([groups[j] for j in sample])
   if np.isfinite(estimate):ci.append(estimate)
  all_rows.append(dict(split=sp,posts=len(groups),annotations=len(g),raters=g.rater.nunique(),multi_rated_posts=n_multi,mean_votes_per_post=float(sizes.mean()),fraction_disagreed=float(disagree),pairwise_exact_agreement=float(pair_good/pair_all) if pair_all else np.nan,krippendorff_alpha_nominal=alpha,alpha_cluster_ci_low=float(np.quantile(ci,.025)) if ci else np.nan,alpha_cluster_ci_high=float(np.quantile(ci,.975)) if ci else np.nan))
  for r,sub in g.groupby('rater',sort=False):
   raters.append(dict(split=sp,rater=r,annotations=len(sub),posts=sub.uid.nunique(),agreement_with_majority=float((sub.vote==sub.label).mean())))
 save_csv(pd.DataFrame(all_rows),'inews_rater_agreement.csv')
 save_csv(pd.DataFrame(raters),'inews_raters.csv')
 save_csv(merged.groupby(['split','vote']).size().rename('n').reset_index(),'inews_individual_vote_distribution.csv')
 postrows=[];pairs=defaultdict(list)
 for (sp,uid),g in merged.groupby(['split','uid']):
  counts=g.vote.value_counts();major=sorted(counts.items(),key=lambda t:(-t[1],t[0]))[0][0]
  postrows.append(dict(split=sp,uid=uid,n_votes=len(g),n_distinct_votes=len(counts),majority_label=major,manifest_label=g.label.iloc[0],majority_fraction=float(counts.iloc[0]/len(g)),entropy=float(-sum((v/len(g))*math.log(v/len(g)) for v in counts))))
  ratings=list(g[['rater','vote']].drop_duplicates('rater').itertuples(index=False,name=None))
  for (ra,va),(rb,vb) in combinations(ratings,2):
   if ra>rb:ra,rb,va,vb=rb,ra,vb,va
   pairs[(sp,ra,rb)].append((va,vb))
 postframe=pd.DataFrame(postrows);save_csv(postframe,'inews_post_vote_consensus.csv')
 if (postframe.majority_label!=postframe.manifest_label).any():
  raise RuntimeError('Individual votes disagree with manifest-majority labels; review iNews preprocessing before interpretation')
 kappas=[]
 for (sp,ra,rb),v in pairs.items():
  if len(v)<10:continue
  va=[t[0] for t in v];vb=[t[1] for t in v];obs=float(np.mean(np.array(va)==np.array(vb)))
  pa=Counter(va);pb=Counter(vb);exp=sum(pa[k]*pb[k] for k in (set(pa)|set(pb)))/len(v)**2
  kappas.append(dict(split=sp,rater_a=ra,rater_b=rb,co_rated_posts=len(v),agreement=obs,cohen_kappa=float((obs-exp)/(1-exp)) if exp<1 else np.nan))
 save_csv(pd.DataFrame(kappas,columns=['split','rater_a','rater_b','co_rated_posts','agreement','cohen_kappa']),'inews_rater_pair_kappa.csv')
 # Evaluate EXISTING test predictions against every human vote; no fitting/retraining.
 test=merged[merged.split=='test'];result=[]
 available=prediction_files('inews','baseline')
 for path in available:
  tag=path.stem.rsplit('_s',1)[0];seed=int(path.stem.rsplit('_s',1)[1]);pred=load_prediction('inews',tag,seed)
  if not pred or 'classes' not in pred:continue
  idx=pred['test_index'];ref=man.iloc[idx];y=pred['test_y'].astype(str)
  if not np.array_equal(ref.label.to_numpy(),y):raise RuntimeError(f'{path}: labels disagree with original manifest')
  cl=pred['classes'].astype(str);yp=cl[pred['test_pred'].argmax(1)]
  vote_map=pd.DataFrame({'uid':ref.uid.values,'pred':yp})
  comparison=test.merge(vote_map,on='uid',how='inner',validate='many_to_one')
  if comparison.empty:continue
  cc=conf_mat(comparison.vote,comparison.pred,cl)
  post_correct=comparison.assign(correct=(comparison.vote==comparison.pred)).groupby('uid').correct.mean()
  result.append(dict(model=tag,seed=seed,posts=len(post_correct),individual_annotations=len(comparison),per_annotation_accuracy=float((comparison.vote==comparison.pred).mean()),post_balanced_accuracy=float(post_correct.mean()),individual_macro_f1=macro_cm(cc),aggregate_macro_f1=macro_cm(conf_mat(y,yp,cl))))
 if not result:raise RuntimeError('No cached iNews predictions found for human-level analysis')
 save_csv(pd.DataFrame(result),'inews_human_level_model_scores.csv')
 log('DONE INEWS')

# ------------------------------------------------------------
# GoodNewsEveryone: role spans; NEVER use original labels as predictors
# ------------------------------------------------------------
def parse_literal(v):
 if not isinstance(v,str) or not v.strip() or v.strip().lower() in ('nan','none','null','na','[]','{}'):return []
 for fn in (json.loads,ast.literal_eval):
  try:return fn(v)
  except (ValueError,SyntaxError,TypeError,OverflowError):pass
 return v

def find_mention(text,phrase):
 phrase=str(phrase).strip().strip('"\'“”’` ')
 if not phrase:return []
 pat=r'\s+'.join(re.escape(x) for x in phrase.split())
 if not pat:return []
 return [(m.start(),m.end()) for m in re.finditer(pat,text,flags=re.I)]

def gold_spans(raw,text):
 """Best-effort reconstruction from raw released GNE role fillers.
 Ambiguous mentions counted and surfaced, no fabricated offset alignment.
 """
 obj=parse_literal(raw);mentions=[];offsets=[]
 def visit(x):
  if x is None:return
  if isinstance(x,dict):
   if 'start' in x and 'end' in x:
    try:offsets.append((int(x['start']),int(x['end'])));return
    except (ValueError,TypeError):pass
   for key in ('text','span','value','surface','words','tokens','spans'):
    if key in x:visit(x[key]);return
   for q in x.values():visit(q)
  elif isinstance(x,(tuple,list)):
   if len(x)==2 and all(isinstance(z,int) for z in x):
    offsets.append(tuple(x));return
   if x and all(isinstance(z,str) for z in x):
    joined=' '.join(x)
    if find_mention(text,joined):mentions.append(joined)
    else:mentions.extend(x)
   else:
    for q in x:visit(q)
  elif isinstance(x,str):
   if x.strip().lower() not in ('none','null','nan','not specified','-','n/a'):
    mentions.append(x)
 visit(obj)
 found=[];not_found=0;ambiguous=0
 for s,e in offsets:
  if 0<=s<e<=len(text):found.append((s,e))
  else:not_found+=1
 for v in mentions:
  hits=find_mention(text,v)
  if not hits:not_found+=1
  else:
   if len(hits)>1:ambiguous+=1
   found.append(hits[0])
 return sorted(set(found)),not_found,ambiguous, bool(mentions or offsets)

def token_ids(text,spans):
 tokens=list(re.finditer(r'\w+|[^\w\s]',text,flags=re.UNICODE));covered=set()
 for i,token in enumerate(tokens):
  if any(token.start()<end and token.end()>start for start,end in spans):covered.add(i)
 return covered

def score_spans(gold,pred,text):
 gt=token_ids(text,gold);pt=token_ids(text,pred);tp=len(gt&pt)
 p=tp/len(pt) if pt else (1. if not gt else 0.)
 r=tp/len(gt) if gt else (1. if not pt else 0.)
 f=2*p*r/(p+r) if p+r else 0.
 return dict(tp=tp,fp=len(pt-gt),fn=len(gt-pt),token_f1=float(f),exact_span=int(sorted(gold)==sorted(pred)),gold_positive=int(bool(gt)),pred_positive=int(bool(pt)))

def load_gne_aligned():
 man=load_manifest('gne');paths=list((RAW/'goodnewseveryone').rglob('gne-release-v1.0.tsv'))
 if not paths:raise FileNotFoundError('GNE raw TSV is required for role annotation analysis')
 raw=pd.read_csv(paths[0],sep='\t',dtype=str,keep_default_na=False)
 if not {'headline','cause','target','experiencer','cue'}.issubset(raw):raise RuntimeError('GNE raw TSV missing annotation columns')
 if len(raw)==len(man) and (raw.headline.astype(str).str.strip().values==man.text.astype(str).str.strip().values).all():
  aligned=raw.reset_index(drop=True).copy();aligned['uid']=man.uid.values;aligned['split']=man.split.values
  return aligned
 # Duplicate headlines: refuse ambiguous text-only mapping, not guess by ID (user's IDs already remapped).
 raw=raw.assign(_norm=raw.headline.astype(str).str.strip());m=man.assign(_norm=man.text.astype(str).str.strip())
 if raw._norm.duplicated().any() or m._norm.duplicated().any():
  raise RuntimeError('GNE raw/manifest order differs and headlines are duplicated; cannot safely align role labels')
 aligned=m[['uid','split','_norm']].merge(raw,on='_norm',how='left',validate='one_to_one')
 if aligned.headline.isna().any():raise RuntimeError('GNE raw headlines fail exact UID alignment')
 return aligned.drop(columns='_norm')

def llm_cache(ds='gne'):
 path=ROOT/'cache'/'llm'/f'{ds}_llm.jsonl';result={}
 for line in path.open(encoding='utf-8'):
  try:
   x=json.loads(line)
   if isinstance(x.get('features'),dict):result[str(x['uid'])]=x['features']
  except (ValueError,KeyError,TypeError):continue
 return result

def parse_roles_response(text):
 text=str(text).strip();ret={}
 start=text.find('{')
 if start>=0:
  for candidate in (text[start:],text[start:].replace("\\'","'")):
   try:
    obj,_=json.JSONDecoder(strict=False).raw_decode(candidate)
    if isinstance(obj,dict):
     for role in ('cause','target','experiencer'):
      if role in obj and isinstance(obj[role],str):ret[role]=obj[role]
     break
   except ValueError:pass
 for role in ('cause','target','experiencer'):
  if role in ret:continue
  m=re.search(r'^\s*["\']?'+role+r'["\']?\s*[:=]\s*(.+)$',text,re.M|re.I)
  if m:ret[role]=m.group(1).strip().rstrip(',').strip('"')
 return ret

def sanitize_role_value(text,val):
 v=str(val).strip().strip('"\'`“” ').rstrip('.')
 if v.lower() in ('none','null','not present','absent','n/a','no explicit span','no explicit experiencer','no explicit target','no explicit cause',''):return 'NONE','explicit_absent'
 hits=find_mention(text,v)
 if len(hits):return text[hits[0][0]:hits[0][1]],('ambiguous' if len(hits)>1 else 'grounded')
 return v,'ungrounded'

def cached_roles():
 p=CACHE/'gne_roles.jsonl';done={}
 if not p.exists():return done
 for line in p.open(encoding='utf-8'):
  try:
   obj=json.loads(line)
   if isinstance(obj.get('roles'),dict):done[obj['uid']]=obj
  except (ValueError,KeyError,TypeError):continue
 return done

def role_generation(gne):
 """New zero-shot role extraction only; old 5000 GNE LLM features unchanged."""
 cache=cached_roles();missing=[r for r in gne.itertuples(index=False) if r.uid not in cache]
 log('GNE ROLE EXTRACTION',len(cache),'cached',len(missing),'pending')
 if not missing:return cache
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 path=str(ROOT/'models'/'qwen2_5_7b_instruct')
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype='auto',device_map='auto').eval()
 CACHE.mkdir(parents=True,exist_ok=True)
 def gen(prompt,max_new=150):
  msgs=[{'role':'user','content':prompt}];rendered=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  enc=tok(rendered,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=1450)
  enc={k:v.to(model.device) for k,v in enc.items()}
  with torch.inference_mode():out=model.generate(input_ids=enc['input_ids'],attention_mask=enc['attention_mask'],max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id)
  return tok.decode(out[0,enc['input_ids'].shape[1]:],skip_special_tokens=True).strip()
 with (CACHE/'gne_roles.jsonl').open('a',encoding='utf-8') as f:
  for i,r in enumerate(missing,1):
   txt=str(r.headline).strip();prompt=(
    'Extract THREE emotion-semantic-role spans from the NEWS HEADLINE only. '\
    'cause: explicit event causing the emotion; target: entity toward which emotion is directed; '\
    'experiencer: entity experiencing the emotion. '\
    'Return a compact JSON object with exactly the keys "cause", "target", "experiencer". '\
    'Each value MUST be a short VERBATIM substring of the headline, or "NONE" if not explicit. '\
    'Do not predict the emotion label; no Markdown.\nHEADLINE: '+txt[:1000])
   obj={};errors=[]
   try:obj=parse_roles_response(gen(prompt))
   except Exception as e:errors.append(repr(e))
   result={};states={}
   for role in ('cause','target','experiencer'):
    value,state=sanitize_role_value(txt,obj.get(role,''))
    if state=='ungrounded' or (role not in obj):
     question=f'Extract only the {role} emotion role from this headline. Answer with ONE exact substring, or NONE. No other text.\nHEADLINE: {txt[:800]}'
     try:value,state=sanitize_role_value(txt,gen(question,80))
     except Exception as e:errors.append(role+':'+repr(e));value='';state='generation_error'
    result[role]=value;states[role]=state
   record=dict(uid=r.uid,roles=result,states=states,errors=errors)
   f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();cache[r.uid]=record
   if i<=3 or i%100==0 or i==len(missing):log('GNE ROLES',i,'/',len(missing),'total',len(cache),'ungrounded',sum(v=='ungrounded' for v in states.values()))
 del model,tok;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return cache

def gne_roles(run_llm=True):
 log('START GNE span-level roles')
 df=load_gne_aligned()
 qa=[]
 for role in ROLES:
  present=aligned=0
  for r in df.itertuples(index=False):
   spans,_,_,has_gold=gold_spans(getattr(r,role),str(r.headline))
   if has_gold:
    present+=1;aligned+=int(bool(spans))
  qa.append(dict(role=role,gold_present=present,gold_alignable=aligned,align_rate=aligned/max(1,present)))
 save_csv(pd.DataFrame(qa),'gne_gold_annotation_alignment_audit.csv')
 for a in qa:
  if a['gold_present']>=100 and a['align_rate']<.70:
   raise RuntimeError(f'GNE released annotation {a["role"]} only {a["align_rate"]:.1%} text-alignable: inspect schema BEFORE zero-shot generation')
 cache=role_generation(df) if run_llm else cached_roles();sem=llm_cache()
 missing_role_ids=set(df.uid.astype(str))-set(cache)
 if missing_role_ids:raise RuntimeError(f'GNE role inference incomplete: {len(missing_role_ids)} UIDs missing; no fabricated negative labels')
 diagnostics=[];scores=[]
 for r in df.itertuples(index=False):
  txt=str(r.headline);uid=str(r.uid);pred_data=cache.get(uid,{});roles=pred_data.get('roles',{});states=pred_data.get('states',{})
  for role in ROLES:
   gold,nf,amb,present=gold_spans(getattr(r,role),txt)
   if role=='cue':
    prediction=sem.get(uid,{}).get('evidence','');state='legacy_llm_evidence'
   else:
    prediction=roles.get(role,'');state=states.get(role,'not_generated')
   if str(prediction).strip().upper() in ('NONE',''):
    predicted=[]
   else:
    predicted=find_mention(txt,prediction)
    predicted=predicted[:1]
   # Keep gold alignment diagnostics; do not score an annotation that is present but unalignable.
   eligible=int(not (present and nf>0 and not gold))
   d=dict(uid=uid,split=r.split,role=role,gold_present=int(present),gold_alignable=int(bool(gold) or not present),gold_missing_mentions=nf,gold_ambiguous_mentions=amb,role_prediction=prediction,pred_status=state,pred_alignable=int(bool(predicted) or str(prediction).upper() in ('NONE','')),scored=eligible)
   if eligible:d.update(score_spans(gold,predicted,txt))
   diagnostics.append(d)
  if uid not in cache and run_llm:raise RuntimeError(f'GNE role generation missing {uid}')
 a=pd.DataFrame(diagnostics);save_csv(a,'gne_role_per_example.csv')
 for (sp,role),g in a.groupby(['split','role']):
  v=g[g.scored==1];tp=v.tp.sum();fp=v.fp.sum();fn=v.fn.sum();mic=2*tp/max(1,2*tp+fp+fn)
  positive=v[v.gold_positive==1]
  scores.append(dict(split=sp,role=role,n_total=len(g),n_evaluable=len(v),n_gold_unalignable=int(len(g)-len(v)),n_ungrounded_predictions=int((g.pred_status=='ungrounded').sum()),micro_token_precision=float(tp/max(1,tp+fp)),micro_token_recall=float(tp/max(1,tp+fn)),micro_token_f1=float(mic),macro_document_span_f1=float(v.token_f1.mean()) if len(v) else np.nan,macro_positive_gold_f1=float(positive.token_f1.mean()) if len(positive) else np.nan,exact_span_accuracy=float(v.exact_span.mean()) if len(v) else np.nan,exact_span_on_positive_gold=float(positive.exact_span.mean()) if len(positive) else np.nan,gold_positive=int(v.gold_positive.sum()),pred_positive=int(v.pred_positive.sum())))
 save_csv(pd.DataFrame(scores),'gne_role_metrics.csv')
 save_csv(a.groupby(['split','role','pred_status']).size().rename('n').reset_index(),'gne_role_status.csv')
 log('DONE GNE roles')

# ------------------------------------------------------------
# GNE intensity: diagnose degenerate label distribution without refitting
# ------------------------------------------------------------
def gne_intensity():
 d=load_manifest('gne');z=d.get('intensity',pd.Series('',index=d.index)).str.lower().str.strip();tr=z[d.split=='train'];te=z[d.split=='test']
 counts=tr.value_counts();major=counts.index[0];mapping={'low':0.,'medium':1.,'high':2.};target=te.map(mapping)
 # unknown weak values intentionally excluded rather than forced to a main category
 ok=target.notna();m=mapping.get(major,np.nan)
 old=ROOT/'results'/'intensity.csv';a=pd.read_csv(old) if old.exists() else pd.DataFrame()
 summary={'n_train':len(tr),'n_val':int((d.split=='val').sum()),'n_test':len(te),'training_majority':major,'train_majority_fraction':float(counts.iloc[0]/len(tr)),'test_majority_fraction':float((te==major).mean()),'test_evaluable':int(ok.sum()),'constant_baseline_MAE':float(np.abs(target[ok]-m).mean()) if np.isfinite(m) and ok.any() else np.nan,'constant_baseline_accuracy':float((target[ok]==m).mean()) if np.isfinite(m) and ok.any() else np.nan,'best_existing_model_MAE':float(pd.to_numeric(a.mae,errors='coerce').min()) if len(a) and 'mae' in a else np.nan,'conclusion':'non-informative severe label imbalance; descriptive/control only'}
 save_csv(pd.DataFrame([summary]),'gne_intensity_degeneracy.csv')
 save_csv(d.groupby(['split','intensity']).size().rename('n').reset_index(),'gne_intensity_counts.csv')
 log('GNE INTENSITY majority',major,'test prevalence',summary['test_majority_fraction'])

# ------------------------------------------------------------
# Pareto: mathematically exact 4D hypervolume of saved fronts
# ------------------------------------------------------------
def hv_exact(points,reference):
 P=np.asarray(points,dtype=np.float64);R=np.asarray(reference,dtype=np.float64)
 if P.ndim!=2 or P.shape[1]!=len(R):raise ValueError('HV dimensions mismatch')
 if not np.isfinite(P).all():raise ValueError('Nonfinite Pareto objectives')
 P=P[np.all(P<R,axis=1)]
 if not len(P):return 0.
 # Remove dominated solutions, avoids exponential sweep on duplicates.
 P=np.unique(P,axis=0)
 keep=np.array([not np.any(np.all(P<=p,axis=1)&np.any(P<p,axis=1)) for p in P]);P=P[keep]
 def vol(Q,ref):
  if len(Q)==0:return 0.
  if Q.shape[1]==1:return max(0.,float(ref[0]-Q[:,0].min()))
  cuts=np.unique(np.r_[Q[:,0],ref[0]]);total=0.
  for left,right in zip(cuts[:-1],cuts[1:]):
   if right<=left:continue
   active=Q[Q[:,0]<=left+1e-12,1:]
   if len(active):total+=(right-left)*vol(active,ref[1:])
  return float(total)
 return vol(P,R)

def pareto_recheck():
 rows=[];ref0=[0.,0.,1.,1.];ref_relaxed=[.05,.05,1.05,1.05]
 src=ROOT/'results'/'pareto.csv'
 if not src.exists():raise FileNotFoundError(src)
 original=pd.read_csv(src)
 for row in original.itertuples(index=False):
  p=ROOT/'results'/'pareto_fronts'/f'{row.dataset}_{row.method}_{int(row.seed)}.csv'
  if not p.exists():raise FileNotFoundError(p)
  df=pd.read_csv(p);cols=['neg_val_f1','neg_worst_recall','source_f1_gap','fraction_active']
  x=df[cols].to_numpy(dtype=float)
  zero_worst=int(np.all(np.isclose(x[:,1],0)))
  raw=hv_exact(x,ref0);rel=hv_exact(x,ref_relaxed)
  rows.append(dict(dataset=row.dataset,method=row.method,seed=int(row.seed),points=len(x),old_MC_hv=row.val_hypervolume_est,exact_original_reference=raw,exact_relaxed_reference=rel,has_positive_original_volume=int(raw>0),all_worst_class_recalls_zero=zero_worst,reference_original=json.dumps(ref0),reference_relaxed=json.dumps(ref_relaxed)))
 result=pd.DataFrame(rows);save_csv(result,'pareto_hypervolume_exact.csv')
 save_csv(result.groupby(['dataset','method'])[['exact_original_reference','exact_relaxed_reference']].agg(['mean','std']).reset_index().set_axis(['dataset','method','original_mean','original_sd','relaxed_mean','relaxed_sd'],axis=1),'pareto_hypervolume_summary.csv')
 log('PARETO exact HV: original reference and explicitly marked relaxed sensitivity check; legacy files untouched')

# ------------------------------------------------------------
# Nonoverlapping statistical extension: bootstrap CIs for unseen slots,
# paired cluster randomization tests with BH over explicitly exploratory post-hoc tests.
# ------------------------------------------------------------
def seed_tag(file):
 stem=file.stem
 if '_s' not in stem:return None
 t,seed=stem.rsplit('_s',1)
 try:return t,int(seed)
 except ValueError:return None

def prediction_info(ds,file):
 ident=seed_tag(file)
 if not ident:return None
 tag,seed=ident
 z=load_prediction(ds,tag,seed)
 if not z or not all(k in z for k in ('classes','test_y','test_pred','test_index')):return None
 cl=[str(x) for x in z['classes']];pred=[cl[j] for j in z['test_pred'].argmax(axis=1)]
 if len(z['test_index'])!=len(pred):raise RuntimeError(f'{file}: test index mismatch')
 y=z['test_y'].astype(str)
 man=load_manifest(ds)
 if tag.startswith('vlm_'):
  # The original engine reindexed a fixed VLM-matched subset BEFORE saving test_index.
  matched=ROOT/'cache'/'llm'/f'{ds}_vlm.jsonl';uids=set()
  if not matched.is_file():raise FileNotFoundError(matched)
  for line in matched.open(encoding='utf-8'):
   try:uids.add(str(json.loads(line)['uid']))
   except (ValueError,KeyError,TypeError):continue
  man=man[man.uid.isin(uids)].reset_index(drop=True)
 if max(z['test_index'])>=len(man):raise RuntimeError(f'{file}: indices are outside the aligned manifest')
 if not np.array_equal(man.iloc[z['test_index']].label.to_numpy(dtype=str),y):raise RuntimeError(f'{file}: gold labels and manifest disagree')
 ids=man.iloc[z['test_index']].group.astype(str).to_numpy()
 fp=hashlib.sha256(np.asarray(pred,dtype='U256').tobytes()+np.asarray(z['test_index']).tobytes()).hexdigest()[:16]
 return dict(dataset=ds,tag=tag,seed=seed,classes=cl,y=y,pred=np.asarray(pred,dtype='U256'),idx=z['test_index'],groups=ids,fingerprint=fp,path=str(file))

def group_conf(p):
 """Compressed per-document TP, true and predicted support (3K, not K²)."""
 classes=p['classes'];ids,inv=np.unique(p['groups'],return_inverse=True);D=len(classes)
 enc={str(c):i for i,c in enumerate(classes)}
 yy=np.array([enc[str(k)] for k in p['y']]);pp=np.array([enc[str(k)] for k in p['pred']])
 g=np.zeros((len(ids),3,D),dtype=np.float32)
 np.add.at(g,(inv,np.ones(len(yy),dtype=int),yy),1)
 np.add.at(g,(inv,np.full(len(pp),2,dtype=int),pp),1)
 match=yy==pp
 np.add.at(g,(inv[match],np.zeros(match.sum(),dtype=int),yy[match]),1)
 return g

def f1_batch(cm):
 cm=np.asarray(cm)
 tp=cm[...,0,:];den=cm[...,1,:]+cm[...,2,:]
 return np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(axis=-1)

def boot_samples(info,B=1200,seed=701):
 rng=np.random.default_rng(seed);gcm=group_conf(info);G=len(gcm)
 weights=rng.multinomial(G,[1/G]*G,size=B).astype(np.float32)
 return f1_batch((weights @ gcm.reshape(G,-1)).reshape(B,3,len(info['classes'])))

def paired_stats(a,b,B=1200,P=1500):
 if a['dataset']!=b['dataset'] or a['seed']!=b['seed']:return None
 if not (np.array_equal(a['idx'],b['idx']) and np.array_equal(a['y'],b['y'])):return None
 if set(a['classes'])!=set(b['classes']) or not np.array_equal(a['groups'],b['groups']):return None
 common=sorted(set(a['classes']))
 a={**a,'classes':common};b={**b,'classes':common}
 ca=group_conf(a);cb=group_conf(b);G=len(ca);K=len(common)
 rng=np.random.default_rng(111+a['seed']);w=rng.multinomial(G,[1/G]*G,size=B).astype(np.float32)
 da=f1_batch((w @ ca.reshape(G,-1)).reshape(B,3,K))
 db=f1_batch((w @ cb.reshape(G,-1)).reshape(B,3,K))
 point=float(f1_batch(ca.sum(0))-f1_batch(cb.sum(0)))
 diff=da-db
 # Paired randomization test: groupwise swapping of *model predictions*,
 # with nonadditive macro-F1 recomputed each permutation.
 permutations=rng.integers(0,2,size=(P,G),dtype=np.int8).astype(np.float32)
 base=ca.sum(0);difference=cb-ca
 candidates=(base.reshape(1,-1)+(permutations @ difference.reshape(G,-1))).reshape(P,3,K)
 other=(ca.sum(0)+cb.sum(0))[None,:,:]-candidates
 null=f1_batch(candidates)-f1_batch(other)
 pval=(1+int(np.count_nonzero(np.abs(null)>=abs(point)-1e-12)))/(P+1)
 return dict(delta_macro_f1=point,ci_low=float(np.quantile(diff,.025)),ci_high=float(np.quantile(diff,.975)),randomization_p=float(pval),n_groups=G,B_boot=B,P_randomization=P)

def stat_extension(B=1200,P=1500):
 log('START STATISTICS; no model retraining')
 legacy=ROOT/'results'/'bootstrap_ci.csv';seen=set()
 if legacy.exists():
  old=pd.read_csv(legacy)
  for x in old.itertuples(index=False):seen.add((str(x.dataset),str(x.model)))
 byds=defaultdict(dict);ci=[];duplicates=[];ci_seen_fingerprints=set()
 CACHE.mkdir(parents=True,exist_ok=True)
 ci_checkpoint=CACHE/f'statistics_ci_B{B}.jsonl'
 pair_checkpoint=CACHE/f'statistics_paired_B{B}_P{P}.jsonl'
 def read_checkpoints(path,key_fields):
  items={}
  if path.exists():
   for line in path.open(encoding='utf-8'):
    try:
     row=json.loads(line)
     items[tuple(str(row[k]) for k in key_fields)]=row
    except (ValueError,KeyError,TypeError):continue
  return items
 cached_ci=read_checkpoints(ci_checkpoint,('dataset','model','seed'))
 cached_pair=read_checkpoints(pair_checkpoint,('dataset','proposed','control','seed'))

 for ds in ('inews','gne','mirage'):
  for suite in ('baseline','ablation','vlm'):
   for path in prediction_files(ds,suite):
    p=prediction_info(ds,path)
    if p is None:continue
    byds[ds][(p['tag'],p['seed'])]=p
    duplicates.append(dict(dataset=ds,model=p['tag'],seed=p['seed'],fingerprint=p['fingerprint']))
    fingerprint_key=(ds,p['tag'],p['fingerprint'])
    # One interval per UNIQUE prediction. Identical seed outputs are not new evidence.
    if (ds,f'{p["tag"]}_s{p["seed"]}') in seen:
     ci_seen_fingerprints.add(fingerprint_key)
     continue
    if fingerprint_key in ci_seen_fingerprints:continue
    ci_seen_fingerprints.add(fingerprint_key)
    key=(ds,p['tag'],str(p['seed']))
    saved=cached_ci.get(key)
    if saved is not None and saved.get('fingerprint')==p['fingerprint']:
     ci.append(saved)
    else:
     vals=boot_samples(p,B=B,seed=70000+int(hashlib.sha1(ds.encode()).hexdigest()[:4],16))
     item=dict(dataset=ds,model=p['tag'],seed=p['seed'],suite=suite,macro_f1=float(f1_batch(group_conf(p).sum(0))),ci_low=float(np.quantile(vals,.025)),ci_high=float(np.quantile(vals,.975)),n_groups=len(set(p['groups'])),bootstrap_repetitions=B,source='new_extension',fingerprint=p['fingerprint'])
     with ci_checkpoint.open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n');f.flush()
     ci.append(item)

    if len(ci)%25==0:log('STAT CI',len(ci),'new rows')
 save_csv(pd.DataFrame(ci),'new_nonoverlapping_bootstrap_ci.csv')
 dup=pd.DataFrame(duplicates);dup['same_predictions_across_seed']=dup.groupby(['dataset','model']).fingerprint.transform('nunique').eq(1)
 save_csv(dup,'seed_prediction_fingerprints.csv')
 # All predeclared baseline vs fixed references plus all ablations vs full; matched VLM vs CLIP baseline.
 pairs=[]
 for ds,items in byds.items():
  for seed in SEEDS:
   refs={'inews':('baseline_tfidf_lr_1','baseline_minilm_mlp_6'),'gne':('baseline_tfidf_lr_1','baseline_minilm_mlp_6'),'mirage':('baseline_tfidf_lr_1','baseline_late_lr_12')}[ds]
   for (tag,s),pred in items.items():
    if s!=seed:continue
    if tag.startswith('baseline_'):
     base=refs[1] if tag!=refs[1] else refs[0]
    elif tag.startswith('ablation_'):
     base='ablation_full_text_lr_0' if ds!='mirage' else 'ablation_late_llm_lr_0'
    elif tag.startswith('vlm_'):
     base='vlm_clipconcat_lr_0'
    else:continue
    if tag==base or (base,seed) not in items:continue
    # Legacy 15 pairs were on a different bootstrap test; skip EXACT duplicate comparisons.
    original_proposed='baseline_full_text_lr_11' if ds!='mirage' else 'baseline_late_llm_rf_13'
    if tag==original_proposed and base=='baseline_tfidf_lr_1':continue
    # Seed 42 is the analysis unit; seed clones are NOT independent replicates.
    if seed!=42:continue
    key=(ds,tag,base,str(seed))
    reference=items[(base,seed)];saved=cached_pair.get(key)
    if saved is not None and saved.get('fingerprint_a')==pred['fingerprint'] and saved.get('fingerprint_b')==reference['fingerprint']:
     pairs.append(saved)
    else:
     ans=paired_stats(pred,reference,B,P)
     if ans:
      item=dict(dataset=ds,suite=tag.split('_')[0],seed=seed,proposed=tag,control=base,fingerprint_a=pred['fingerprint'],fingerprint_b=reference['fingerprint'],**ans)
      with pair_checkpoint.open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n');f.flush()
      pairs.append(item)

  log('STAT PAIRS',ds,len(pairs))
 if not pairs:raise RuntimeError('No aligned cached predictions available for paired tests')
 a=pd.DataFrame(pairs);p=a.randomization_p.to_numpy();order=np.argsort(p);adjust=np.empty(len(p));running=1.
 for k in range(len(p)-1,-1,-1):
  j=order[k];running=min(running,p[j]*len(p)/(k+1));adjust[j]=running
 a['bh_q_all_exploratory']=adjust;save_csv(a,'paired_randomization_BH_exploratory.csv')
 log('DONE STATISTICS',len(ci),'new CIs',len(a),'new pairwise tests')

def extension_figures():
 """New-only manuscript figures; cream background, viridis, readable typography."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 cmap=plt.get_cmap('viridis')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':16,
  'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
  'figure.facecolor':'#faf8f3','axes.facecolor':'#faf8f3',
  'text.color':'#303442','axes.labelcolor':'#303442',
  'xtick.color':'#303442','ytick.color':'#303442',
  'axes.edgecolor':'#848999','grid.color':'#c9c9c2',
  'savefig.facecolor':'#faf8f3','pdf.fonttype':42})
 directory=OUT/'figures';directory.mkdir(parents=True,exist_ok=True)
 def finish(fig,stem):
  fig.savefig(directory/(stem+'.pdf'),bbox_inches='tight',dpi=220)
  fig.savefig(directory/(stem+'.png'),bbox_inches='tight',dpi=180)
  plt.close(fig)
 # Annotation agreement by split
 df=pd.read_csv(OUT/'inews_rater_agreement.csv');x=np.arange(len(df))
 fig,ax=plt.subplots(figsize=(9,5.5));ax.bar(x-.18,df.pairwise_exact_agreement,width=.35,color=cmap(.48),alpha=.82,label='Pairwise agreement')
 ax.bar(x+.18,df.krippendorff_alpha_nominal,width=.35,color=cmap(.77),alpha=.82,label='Nominal Krippendorff alpha')
 ax.set_xticks(x,df.split.str.upper());ax.set_ylabel('Agreement coefficient');ax.set_title('Human annotation agreement — iNews')
 ax.axhline(0,color='#555b63',lw=.8);ax.grid(axis='y',alpha=.45);ax.legend(frameon=False,loc='best');fig.tight_layout();finish(fig,'01_inews_annotators')
 # four role extraction, split-specific score matrix
 df=pd.read_csv(OUT/'gne_role_metrics.csv');splits=['train','val','test'];roles=['cue','cause','target','experiencer']
 a=np.full((len(roles),len(splits)),np.nan)
 for i,role in enumerate(roles):
  for j,split in enumerate(splits):
   v=df[(df.role==role)&(df.split==split)]
   if len(v):a[i,j]=float(v.iloc[0].micro_token_f1)
 fig,ax=plt.subplots(figsize=(9,5.5));heat=ax.imshow(np.ma.masked_invalid(a),cmap='viridis',vmin=0,vmax=1,aspect='auto')
 ax.set_xticks(range(3),[s.upper() for s in splits]);ax.set_yticks(range(4),[r.capitalize() for r in roles]);ax.set_title('Emotion-role span evaluation — GNE')
 for i in range(4):
  for j in range(3):
   if np.isfinite(a[i,j]):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='#202735',fontweight='bold',bbox={'facecolor':'#faf8f3','edgecolor':'none','alpha':.76,'pad':2})
 fig.colorbar(heat,ax=ax,label='Token-level micro-F1',shrink=.82);fig.tight_layout();finish(fig,'02_gne_role_spans')
 # Validation Pareto hypervolume under relaxed, clearly labeled sensitivity reference
 df=pd.read_csv(OUT/'pareto_hypervolume_exact.csv');tab=df.groupby(['dataset','method']).exact_relaxed_reference.mean().unstack()
 fig,ax=plt.subplots(figsize=(9.5,5.5));ms=sorted(tab.columns);width=.23;x=np.arange(len(tab))
 for j,method in enumerate(ms):ax.bar(x+(j-(len(ms)-1)/2)*width,tab[method].fillna(0).to_numpy(),width,color=cmap(.23+.6*j/max(1,len(ms)-1)),alpha=.83,label=method.upper())
 ax.set_xticks(x,[str(q).upper() for q in tab.index]);ax.set_ylabel('Exact dominated hypervolume');ax.set_title('Pareto front sensitivity: relaxed reference');ax.grid(axis='y',alpha=.4);ax.legend(frameon=False);fig.tight_layout();finish(fig,'03_pareto_hypervolume_sensitivity')
 # Confidence intervals of differences from paired randomization protocol
 df=pd.read_csv(OUT/'paired_randomization_BH_exploratory.csv');df=df.sort_values('bh_q_all_exploratory').head(22).iloc[::-1]
 labels=(df.dataset+' / '+df.proposed.str.replace('baseline_','',regex=False)).str.slice(0,46)
 fig,ax=plt.subplots(figsize=(12,max(6,len(df)*.36)));y=np.arange(len(df))
 for i,r in enumerate(df.itertuples(index=False)):
  c=cmap(.75 if r.bh_q_all_exploratory<.05 else .33)
  ax.plot([r.ci_low,r.ci_high],[i,i],lw=2.3,color=c,alpha=.85)
  ax.plot([r.delta_macro_f1],[i],marker='o',ms=7,color=c)
 ax.set_yticks(y,labels);ax.axvline(0,color='#505660',lw=1,linestyle='--');ax.set_xlabel('Paired macro-F1 difference with 95% cluster-bootstrap CI');ax.set_title('Exploratory paired comparisons (post hoc)');ax.grid(axis='x',alpha=.4);fig.tight_layout();finish(fig,'04_paired_differences')
 log('NEW FIGURES',directory,'4 PDF + 4 PNG')


def sanity():
 m=load_manifest('gne');s=load_manifest('inews');log('PREFLIGHT','iNews',len(s),'GNE',len(m))
 for ds in ('inews','gne','mirage'):
  if not prediction_files(ds):raise RuntimeError(f'No existing predictions for {ds}; refusing to retrain old models')
 if not (ROOT/'results'/'pareto.csv').exists():raise RuntimeError('Original Pareto results missing')
 if not (ROOT/'results'/'intensity.csv').exists():raise RuntimeError('Original GNE intensity missing')

def summary():
 names=['inews_rater_agreement.csv','inews_human_level_model_scores.csv','gne_role_metrics.csv','gne_intensity_degeneracy.csv','pareto_hypervolume_exact.csv','new_nonoverlapping_bootstrap_ci.csv','paired_randomization_BH_exploratory.csv']
 rows=[]
 for n in names:
  p=OUT/n;count=(len(pd.read_csv(p)) if p.exists() else 0)
  rows.append(dict(file=n,exists=p.exists(),rows=count))
 for r in rows:log('RESULT',r)
 save_csv(pd.DataFrame(rows),'extension_progress.csv')
 if not all(r['exists'] for r in rows):raise RuntimeError('Extension incomplete; check missing files')

def run(args):
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 # Linux flock: prevent two independent GPU jobs writing to the same JSONL.
 import fcntl
 lock=(CACHE/'scientific_extension.lock').open('a')
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Another Phase2 scientific extension process is already running')
 sanity()
 stages=[('inews',inews_humans),('gne_intensity',gne_intensity),('gne_roles',lambda:gne_roles(run_llm=not args.no_llm)),('pareto',pareto_recheck),('statistics',lambda:stat_extension(args.bootstrap,args.permutations)),('figures',extension_figures)]
 for stage,func in stages:
  if args.only and stage not in args.only:continue
  # Safe idempotent cache; avoid skipping an unfinished long GNE role generation.
  log('STAGE START',stage)
  func()
  log('STAGE DONE',stage)
 if not args.only:summary()

def self_test():
 assert abs(hv_exact(np.array([[0.2,.3]]),[1.,1.])-.56)<1e-10
 assert abs(hv_exact(np.array([[.2,.3],[.6,.1]]),[1.,1.])-(.8*.7+.4*.2))<1e-10
 assert parse_roles_response('{"cause":"A", "target":"B", "experiencer":"NONE"}')['target']=='B'
 assert gold_spans('["foo bar"]','Oh foo bar!')[0]==[(3,10)]
 assert score_spans([(0,3)],[(0,3)],'foo bar')['token_f1']==1
 assert math.isclose(macro_cm(np.array([[3,1],[2,4]])),.7,abs_tol=.06)
 x={'classes':['0','1'],'y':np.array(['0','0','1','1']),'pred':np.array(['0','1','1','1']),'groups':np.array(['a','b','c','d'])}
 assert len(boot_samples(x,20))==20
 log('SELF-TEST PASS')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--only',nargs='*',choices=['inews','gne_intensity','gne_roles','pareto','statistics','figures']);ap.add_argument('--bootstrap',type=int,default=1200);ap.add_argument('--permutations',type=int,default=1500);ap.add_argument('--no-llm',action='store_true');ap.add_argument('--self-test',action='store_true')
 args=ap.parse_args()
 try:
  if args.self_test:self_test()
  else:run(args)
 except KeyboardInterrupt:log('INTERRUPTED — original files untouched');sys.exit(130)
 except Exception as exc:log('FAILED',repr(exc));raise
PY_EXTENSION_SOURCE
PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ -x "$PY" ] && [ "$(sha256sum scripts/phase2_scientific_extension.py | awk '{print $1}')" = "ab0bc61802c8798e456a9769788c821543fb9090ba6f37c4b468375b82d237bf" ]; then
  if "$PY" -m py_compile scripts/phase2_scientific_extension.py && "$PY" -u scripts/phase2_scientific_extension.py --self-test; then
    echo "========== PHASE 2: SCIENTIFIC EXTENSION =========="
    ( set -o pipefail; "$PY" -u scripts/phase2_scientific_extension.py 2>&1 | tee -a logs/phase2_scientific_extension.log )
    status=$?
    if [ "$status" -eq 0 ]; then
      echo "[SUCCESS] Results: /home/tahiti/NewsEmo/results/phase2_extension/"
    else
      echo "[STOP] Status: $status. All existing results and caches preserved."
    fi
  else
    echo "[STOP] Syntax or self-test failed."
  fi
else
  echo "[STOP] Python missing or script checksum mismatch."
fi
fi
echo "[TERMINAL ACTIVE]"
