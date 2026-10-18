#!/usr/bin/env python3
"""NewsEmo Phase 2 reproducible experiment engine. No auto-installs / downloads."""
import os, sys, io, re, gc, csv, json, math, time, glob, random, hashlib, zipfile, argparse, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
ROOT=Path('/home/tahiti/NewsEmo')
EMOTIONS=['anger','disgust','fear','joy','sadness','surprise']
SEEDS=[42,43,44,45,46]

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def jsonsave(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');q.replace(p)
def hval(x): return hashlib.sha1(str(x).encode()).hexdigest()[:16]
def append_table(path,row):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 new=pd.DataFrame([row])
 if path.exists():
  try:new=pd.concat([pd.read_csv(path),new],ignore_index=True,sort=False)
  except pd.errors.EmptyDataError:pass
 new.to_csv(path,index=False)
def dfpath(name): return ROOT/'cache'/'manifests'/f'{name}.csv'
def read_df(name):
 p=dfpath(name)
 if not p.exists(): raise FileNotFoundError(f'{p}: run prepare first')
 return pd.read_csv(p,keep_default_na=False)
def save_df(name,df):
 p=dfpath(name);p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False);log('SAVED',name,len(df),p)
def mkrow(uid,text,label,split,source='',image='',group='',**kw):
 return dict(uid=str(uid),text=str(text),label=str(label),split=str(split),source=str(source),image=str(image),group=str(group or uid),**kw)
def split_groups(df,seed=42):
 from sklearn.model_selection import GroupShuffleSplit
 groups=df['group'].astype(str)
 gss=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=seed)
 trv,te=next(gss.split(df,groups=groups))
 sub=df.iloc[trv]; gss2=GroupShuffleSplit(n_splits=1,test_size=.176470588,random_state=seed+1)
 tr,va=next(gss2.split(sub,groups=sub['group'].astype(str)))
 df=df.copy();df['split']='test';df.loc[df.index[trv[tr]],'split']='train';df.loc[df.index[trv[va]],'split']='val'
 return df

def prepare(limit=0,force=False):
 d=ROOT/'datasets'; (ROOT/'cache'/'images').mkdir(parents=True,exist_ok=True)
 # iNews: no repeated human ratings as independent news; majority vote per post
 if force or not dfpath('inews').exists():
  parts=[]
  for f,s in [('train.csv','train'),('dev.csv','val'),('test_cold_start_public.csv','test')]:
   a=pd.read_csv(d/'inews_public'/f,low_memory=False)
   a['split']=s;parts.append(a)
  a=pd.concat(parts,ignore_index=True);a['Post_ID']=a.Post_ID.astype(str)
  # Official cold-start test kept untouched; exclude other repeats from fit/validation
  testids=set(a.loc[a.split=='test','Post_ID']);valids=set(a.loc[a.split=='val','Post_ID'])
  a=a[~((a.split!='test') & a.Post_ID.isin(testids))]
  a=a[~((a.split=='train') & a.Post_ID.isin(valids))]
  rows=[]
  for (sp,pid),g in a.groupby(['split','Post_ID'],sort=False):
   votes=g.Discrete.dropna().astype(str).str.strip();votes=votes[votes.str.len()>0]
   if len(votes)==0:continue
   label=sorted(votes.value_counts().items(),key=lambda t:(-t[1],t[0]))[0][0]
   rr=g.iloc[0];text=(' '.join([str(rr.get('Message','') or ''),str(rr.get('Link.Text','') or '')])).strip()
   if len(text)<5:continue
   vad={}
   for v in ('V','A','D'):
    vv=pd.to_numeric(g[v],errors='coerce') if v in g else pd.Series(dtype=float)
    vad[v]=float(vv.mean()) if len(vv) and vv.notna().any() else np.nan
   rows.append(mkrow(pid,text,label,sp,str(rr.get('Page.Name','unknown')),group=pid,n_raters=len(votes),**vad))
  f=pd.DataFrame(rows).drop_duplicates('uid');save_df('inews',f.head(limit) if limit else f)
 # GoodNewsEveryone: source + named entities only as targets, NEVER as inputs
 if force or not dfpath('gne').exists():
  p=next((d/'goodnewseveryone').rglob('gne-release-v1.0.tsv'))
  a=pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False)
  rows=[]
  for _,r in a.iterrows():
   text=r['headline'].strip();lab=r['dominant_emotion'].strip().lower()
   if not text or not lab:continue
   gid=hval(re.sub(r'\W+',' ',text.lower()).strip())
   rows.append(mkrow(r['id'],text,lab,'pending',r.get('source','unknown'),group=gid,intensity=r.get('intensity',''),cue=r.get('cue',''),cause=r.get('cause',''),reader_emotions=r.get('reader_emotions','')))
  f=split_groups(pd.DataFrame(rows));save_df('gne',f.head(limit) if limit else f)
 # SemEval: external human annotations only, zero-shot evaluation; 6 raters
 if force or not dfpath('semeval').exists():
  import xml.etree.ElementTree as ET
  base=d/'semeval2007_affective_text';xml=ET.parse(base/'affectivetext_test.xml').getroot()
  titles={int(e.attrib['id']):str(e.text or '').strip() for e in xml}
  records=[]
  for p in sorted((base/'Annotations').glob('*.txt')):
   for line in p.read_text(errors='replace').splitlines():
    x=line.strip().split()
    if len(x)<8: continue
    try: vals=[float(v) for v in x[:8]]
    except ValueError:continue
    records.append([int(vals[0]),*vals[1:]])
  tab=pd.DataFrame(records,columns=['id',*EMOTIONS,'valence']).groupby('id').mean()
  rows=[]
  for iid,text in titles.items():
   if iid not in tab.index:continue
   r=mkrow(iid,text,'external','external','semeval',group=iid)
   for k in [*EMOTIONS,'valence']:r[k]=float(tab.loc[iid,k])
   rows.append(r)
  save_df('semeval',pd.DataFrame(rows).head(limit) if limit else pd.DataFrame(rows))
 # Mirage: write images from Parquet ONE FILE / BATCH AT A TIME. Ignore duplicate unlabeled generic test files.
 if force or not dfpath('mirage').exists():
  from PIL import Image
  import pyarrow.parquet as pq
  base=d/'mirage_news'/'data';imgs=ROOT/'cache'/'images'/'mirage';imgs.mkdir(parents=True,exist_ok=True)
  specs=[('train',sorted(base.glob('train-*.parquet'))),('val',sorted(base.glob('validation-*.parquet')))]
  for i in range(1,6):specs.append((f'test{i}',sorted(base.glob(f'test{i}_*.parquet'))))
  rows=[]
  for sp,files in specs:
   if not files:raise FileNotFoundError(f'MiRAGeNews missing split {sp}')
   index=0
   for p in files:
    pf=pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=48,columns=['text','label','image']):
     for rec in batch.to_pylist():
      if limit and index>=limit:break
      uid=f'{sp}_{index:05d}';out=imgs/f'{uid}.jpg'
      if not out.is_file():
       dat=rec['image'];bits=dat.get('bytes') if isinstance(dat,dict) else None
       if bits:
        try:Image.open(io.BytesIO(bits)).convert('RGB').save(out,'JPEG',quality=89)
        except Exception as ex:log('BAD IMAGE',uid,ex)
      lab=rec['label'];lab=str(int(lab)) if lab is not None else ''
      source={'train':'NYT_MJ','val':'NYT_MJ','test1':'NYT_MJ','test2':'BBC_DALLE','test3':'CNN_DALLE','test4':'BBC_SDXL','test5':'CNN_SDXL'}[sp]
      rows.append(mkrow(uid,rec['text'],lab,sp,source,str(out) if out.exists() else '',group=uid));index+=1
     if limit and index>=limit:break
   log('MIRAGE',sp,index)
  save_df('mirage',pd.DataFrame(rows))
 # NewsCLIPpings: 6925 valid paired samples, test ONLY
 if force or not dfpath('clips').exists():
  zfile=d/'newsclippings_ready'/'test_dataset_newsclipping.zip';imgs=ROOT/'cache'/'images'/'clips';imgs.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(zfile) as z:
   allnames=z.namelist();lookup={'origin/'+v.split('/origin/',1)[1]:v for v in allnames if '/origin/' in v and v.lower().endswith(('.jpg','.jpeg','.png','.webp'))}
   def load_json(name):
    with z.open(next(v for v in allnames if v.endswith('/'+name))) as fp:return json.load(fp)
   ann=load_json('news_clippings_test.json')['annotations'];meta=load_json('visual_news_test.json');rows=[]
   for j,r in enumerate(ann):
    if limit and len(rows)>=limit:break
    c=meta.get(str(r['id']));im=meta.get(str(r['image_id']))
    if not c or not im:continue
    caption=str(c.get('caption') or c.get('text') or '').strip();raw=str(im.get('image_path','')).replace('\\','/')
    if not caption or 'origin/' not in raw:continue
    key='origin/'+raw.split('origin/',1)[1];member=lookup.get(key)
    if not member:continue
    out=imgs/f'{r["image_id"]}{Path(key).suffix.lower()}'
    if not out.exists():
     try:
      with z.open(member) as src,out.open('wb') as dst:
       import shutil;shutil.copyfileobj(src,dst)
     except Exception as ex:log('BAD CLIPS',j,ex);continue
    rows.append(mkrow(f'clip_{j}',caption,int(bool(r['falsified'])),'external',str(r.get('source_dataset','unknown')),str(out),group=str(r['id']),image_id=r['image_id'],caption_id=r['id']))
  save_df('clips',pd.DataFrame(rows))
 log('PREPARE DONE')

def text_stats(df):
 a=[]
 for t in df.text.astype(str):
  words=re.findall(r"\b\w+\b",t);n=max(1,len(words));a.append([len(t),len(words),len(set(w.lower() for w in words))/n,t.count('.'),t.count('!'),t.count('?'),sum(c.isupper() for c in t)/max(1,len(t)),sum(c.isdigit() for c in t)/max(1,len(t)),len(re.findall(r'\"|“|”',t)),len(re.findall(r'\b(?:said|according|reported|claims?)\b',t.lower()))])
 return np.asarray(a,np.float32)
def featpath(ds,kind):return ROOT/'cache'/'features'/f'{ds}_{kind}.npy'
def extract_stats(ds):
 p=featpath(ds,'stats');p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():np.save(p,text_stats(read_df(ds)))
 log('STATS',ds,np.load(p,mmap_mode='r').shape)

def dense_extract(ds,kind,batch=12):
 import torch
 from transformers import AutoTokenizer,AutoModel,CLIPModel,CLIPProcessor,AutoImageProcessor
 from PIL import Image
 df=read_df(ds);p=featpath(ds,kind);p.parent.mkdir(parents=True,exist_ok=True)
 done=p.with_suffix('.progress.json');start=0
 dev='cuda' if torch.cuda.is_available() else 'cpu';dtype=torch.float16 if dev=='cuda' else torch.float32
 paths={'minilm':'minilm_l6_v2','cliptext':'clip_vit_b32','clipimage':'clip_vit_b32','cliptextL':'clip_vit_l14','clipimageL':'clip_vit_l14','dinov2':'dinov2_small'}
 path=str(ROOT/'models'/paths[kind]);log('LOAD',kind,path,dev)
 if kind=='minilm':
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModel.from_pretrained(path,local_files_only=True).to(dev).eval();dim=int(model.config.hidden_size)
 elif kind in ('cliptext','clipimage','cliptextL','clipimageL'):
  tok=CLIPProcessor.from_pretrained(path,local_files_only=True);model=CLIPModel.from_pretrained(path,local_files_only=True).to(dev).eval();dim=int(model.config.projection_dim)
 else:
  tok=AutoImageProcessor.from_pretrained(path,local_files_only=True);model=AutoModel.from_pretrained(path,local_files_only=True).to(dev).eval();dim=int(model.config.hidden_size)
 if p.exists() and done.exists():
  prog=json.loads(done.read_text());old=np.load(p,mmap_mode='r')
  if tuple(old.shape)==(len(df),dim):start=int(prog.get('next',0))
  else:start=0
 if start==0:arr=np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(len(df),dim))
 else:arr=np.lib.format.open_memmap(p,mode='r+')
 for pos in range(start,len(df),batch):
  subs=df.iloc[pos:pos+batch];indices=[];images=[]
  if kind in ('clipimage','clipimageL','dinov2'):
   for k,fp in enumerate(subs.image.astype(str).tolist()):
    try:
     if not fp:continue
     with Image.open(fp) as pic:images.append(pic.convert('RGB').copy());indices.append(k)
    except Exception:continue
   output=np.zeros((len(subs),dim),np.float32)
   if images:
    if kind in ('clipimage','clipimageL'):
     inp=tok(images=images,return_tensors='pt',padding=True).to(dev)
     with torch.inference_mode():z=model.get_image_features(**inp)
    else:
     inp=tok(images=images,return_tensors='pt').to(dev)
     with torch.inference_mode():z=model(**inp).last_hidden_state[:,0,:]
    if not isinstance(z,torch.Tensor):z=z.pooler_output
    output[np.asarray(indices)]=z.detach().float().cpu().numpy()
  else:
   texts=subs.text.astype(str).tolist()
   if kind in ('cliptext','cliptextL'):
    inp=tok(text=texts,return_tensors='pt',padding=True,truncation=True,max_length=77).to(dev)
    with torch.inference_mode():z=model.get_text_features(**inp)
   else:
    inp=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=256).to(dev)
    with torch.inference_mode():
     h=model(**inp).last_hidden_state;mask=inp['attention_mask'].unsqueeze(-1)
     z=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
   if not isinstance(z,torch.Tensor):z=z.pooler_output
   output=z.detach().float().cpu().numpy()
  arr[pos:pos+len(subs)]=output;arr.flush()
  jsonsave(done,{'next':pos+len(subs),'rows':len(df),'dim':dim})
  if pos%(batch*25)==0:log('EXTRACT',kind,ds,pos+len(subs),'/',len(df))
 log('COMPLETE',kind,ds);del model;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None

# LLM features never include emotion label or direct response. Separate zero-shot emotion experiment.
NUMKEYS=['uncertainty','urgency','causal_explicitness','attribution_strength','quotation_density','sensational_language']
TEXTKEYS=['frame','stance','focus','evidence']
VNUM=['visual_salience','people_presence','scene_uncertainty','image_text_alignment','image_text_complementarity','visual_emphasis']
VTEXT=['scene','objects','activity','visual_evidence']
def parse_json(s):
 try:return json.loads(s)
 except Exception:
  m=re.search(r'\{.*\}',s,re.S)
  if m:
   try:return json.loads(m.group())
   except Exception:pass
 return None
def llm_extract(ds,kind='llm',limit=0):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,AutoProcessor
 from PIL import Image
 df=read_df(ds);out=ROOT/'cache'/'llm'/f'{ds}_{kind}.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
 done={}
 if out.exists():
  for line in out.read_text(encoding='utf-8').splitlines():
   try:r=json.loads(line);done[str(r['uid'])]=r
   except Exception:pass
 name='qwen2_5_7b_instruct' if kind=='llm' else 'qwen2_5_vl_7b';path=str(ROOT/'models'/name)
 dev='cuda' if torch.cuda.is_available() else 'cpu';log('LOAD LLM',name,dev)
 if kind=='llm':
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(path,torch_dtype='auto',device_map='auto',local_files_only=True).eval()
 else:
  from transformers import Qwen2_5_VLForConditionalGeneration
  tok=AutoProcessor.from_pretrained(path,local_files_only=True)
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,torch_dtype='auto',device_map='auto',local_files_only=True).eval()
 candidates=df
 if kind=='vlm':
  # Precommitted matched subset, independent of labels and predictions
  quotas={'train':2000,'val':500,**{f'test{k}':100 for k in range(1,6)}} if ds=='mirage' else {'external':1000}
  choices=[]
  for split,quota in quotas.items():
   sub=df[(df.split==split)&df.image.astype(str).map(lambda p:bool(p) and Path(p).is_file())].copy()
   sub=sub.sort_values('uid',key=lambda s:s.map(hval)).head(quota)
   choices.append(sub)
  candidates=pd.concat(choices,ignore_index=True)
 if limit:candidates=candidates.head(limit)
 keys=NUMKEYS if kind=='llm' else VNUM;strings=TEXTKEYS if kind=='llm' else VTEXT
 for j,r in enumerate(candidates.itertuples(index=False),1):
  uid=str(r.uid)
  if uid in done and 'features' in done[uid]:continue
  request=('Analyze the observable writing style of this news text. Never predict emotions, reader response, news authenticity or a target class. '
   'Return valid JSON only, with numeric fields '+', '.join(keys)+' in [0,1], and short string fields '+', '.join(strings)+'. '
   'Evidence must contain a verbatim short supporting substring when available. Text: '+str(r.text)[:2700]) if kind=='llm' else (
   'Analyze the supplied news image and caption, without predicting fake/real or any ground-truth label. '
   'Return JSON with numeric fields '+', '.join(keys)+' in [0,1], and short string fields '+', '.join(strings)+'. '
   'Caption: '+str(r.text)[:550])
  try:
   if kind=='llm':
    msgs=[{'role':'user','content':request}];inp=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():res=model.generate(inp,max_new_tokens=320,do_sample=False,pad_token_id=tok.eos_token_id)
    ans=tok.decode(res[0][inp.shape[1]:],skip_special_tokens=True)
   else:
    image=Image.open(str(r.image)).convert('RGB');msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':request}]}]
    prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(text=[prompt],images=[image],padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():res=model.generate(**inp,max_new_tokens=320,do_sample=False)
    ans=tok.batch_decode(res[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0]
   obj=parse_json(ans)
   if not isinstance(obj,dict):
    log('BAD JSON RETRY',kind,uid,ans[:120])
    try:
     if kind=='llm':
      with torch.inference_mode():
       res=model.generate(inp,max_new_tokens=640,do_sample=False,pad_token_id=tok.eos_token_id)
      ans=tok.decode(res[0][inp.shape[1]:],skip_special_tokens=True)
     else:
      with torch.inference_mode():
       res=model.generate(**inp,max_new_tokens=640,do_sample=False)
      ans=tok.batch_decode(res[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0]
     obj=parse_json(ans)
    except Exception as ex_retry:
     log('JSON RETRY ERROR',kind,uid,repr(ex_retry))
   if not isinstance(obj,dict):
    log('BAD JSON SKIPPED',kind,uid,ans[:160])
    continue
   rec={'uid':uid,'features':obj}
   with out.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
   done[uid]=rec
  except Exception as ex:log('ERROR LLM',kind,uid,repr(ex))
  if j%50==0:log('LLM',kind,ds,j,'/',len(candidates),'cached',len(done))
 log('LLM COMPLETE',kind,ds,len(done),'/',len(candidates));del model;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None

def sem_features(ds,kind='llm',drop=()):
 from sklearn.feature_extraction.text import HashingVectorizer
 df=read_df(ds);p=ROOT/'cache'/'llm'/f'{ds}_{kind}.jsonl';mapping={}
 if not p.exists():raise FileNotFoundError(f'No {kind} output for {ds}: {p}')
 for line in p.open(encoding='utf-8'):
  try:r=json.loads(line);mapping[str(r['uid'])]=r['features']
  except Exception:pass
 if not mapping:raise RuntimeError(f'Empty {kind} cache: {p}')
 if kind=='llm':
  dfcheck=read_df(ds);unavailable=set(dfcheck.uid.astype(str))-set(mapping)
  if unavailable:raise RuntimeError(f'Incomplete LLM cache {ds}: {len(unavailable)} of {len(dfcheck)} missing')
 nums=NUMKEYS if kind=='llm' else VNUM;strings=TEXTKEYS if kind=='llm' else VTEXT
 m=np.zeros((len(df),len(nums)),np.float32);txt=[]
 for i,uid in enumerate(df.uid.astype(str)):
  rec=mapping.get(uid,{})
  for j,k in enumerate(nums):
   try:m[i,j]=float(np.clip(float(rec.get(k,0)),0,1)) if k not in drop else 0
   except (ValueError,TypeError):pass
  txt.append(' '.join(str(rec.get(k,'')) for k in strings if k not in drop))
 hv=HashingVectorizer(n_features=128,alternate_sign=False,norm='l2',ngram_range=(1,2))
 z=hv.transform(txt).astype(np.float32).toarray()
 return np.column_stack([m,z]).astype(np.float32)

def features(ds,kind,drop=()):
 if kind in ('llm','vlm'):return sem_features(ds,kind,drop)
 p=featpath(ds,kind)
 if not p.is_file():raise FileNotFoundError(f'Missing {kind} features for {ds}. Run extract first: {p}')
 return np.load(p,mmap_mode='r').astype(np.float32)
def blocks(ds):
 b={'stats':features(ds,'stats')}
 for k in ('minilm','cliptext','clipimage','cliptextL','clipimageL','dinov2','llm','vlm'):
  try:b[k]=features(ds,k)
  except FileNotFoundError:
   pass
 return b

def selector(ds,variant,b,seed=42):
 from sklearn.preprocessing import normalize
 rng=np.random.default_rng(seed)
 def cat(*keys):
  missing=[k for k in keys if k not in b]
  if missing:raise FileNotFoundError(f'{ds} {variant} missing feature blocks {missing}')
  return np.column_stack([np.asarray(b[k]) for k in keys]).astype(np.float32)
 if variant=='tfidf':return None
 if variant=='majority':return np.zeros((len(next(iter(b.values()))),1),np.float32)
 if variant=='stats':return cat('stats')
 if variant=='minilm':return cat('minilm')
 if variant=='llm':return cat('llm')
 if variant=='mini_stats':return cat('minilm','stats')
 if variant=='tf_llm':return 'tf_llm'
 if variant=='mini_llm':return cat('minilm','llm')
 if variant=='full_text':return cat('minilm','stats','llm')
 if variant=='cliptext':return cat('cliptext')
 if variant=='clipimage':return cat('clipimage')
 if variant=='dinov2':return cat('dinov2')
 if variant=='clipconcatL':return cat('cliptextL','clipimageL')
 if variant=='similarity':
  a=normalize(cat('cliptext'));z=normalize(cat('clipimage'));return (a*z).sum(axis=1,keepdims=True)
 if variant=='clipconcat':return cat('cliptext','clipimage')
 if variant=='dino_text':return cat('minilm','dinov2')
 if variant=='clip_llm_text':return cat('cliptext','llm')
 if variant=='clip_llm':return cat('cliptext','clipimage','llm')
 if variant=='late':return cat('stats','minilm','cliptext','clipimage','dinov2') if ds=='mirage' else cat('stats','minilm','llm')
 if variant=='late_llm':return cat('stats','minilm','cliptext','clipimage','dinov2','llm')
 if variant=='vlm':return cat('cliptext','clipimage','vlm')
 if variant=='vlm_llm':return cat('cliptext','clipimage','llm','vlm')
 if variant=='no_stats':return cat('minilm','llm')
 if variant=='no_llm':return cat('minilm','stats')
 if variant=='no_mini':return cat('stats','llm')
 if variant=='no_cliptext':return cat('clipimage','dinov2','llm')
 if variant=='no_clipimage':return cat('cliptext','dinov2','llm')
 if variant=='no_dino':return cat('cliptext','clipimage','llm')
 if variant=='no_sem':return cat('cliptext','clipimage','dinov2')
 if variant=='no_visual':return cat('cliptext','llm')
 if variant=='no_text':return cat('clipimage','dinov2')
 if variant=='no_similarity':return cat('cliptext','clipimage','dinov2','llm')
 if variant=='shuffled_image':
  # Shuffle within each split; no cross-split image transfer.
  img=np.asarray(b['clipimage']).copy();splits=read_df(ds).split.astype(str).to_numpy();idx=np.arange(len(img))
  for sp in np.unique(splits):
   where=np.flatnonzero(splits==sp);idx[where]=rng.permutation(where)
  return np.column_stack([b['cliptext'],img[idx],b['llm']]).astype(np.float32)
 if variant.startswith('no_'):
  key=variant[3:]
  if key in NUMKEYS+TEXTKEYS:
   return np.column_stack([b['minilm'],b['stats'],sem_features(ds,drop=(key,))]).astype(np.float32)
 raise ValueError(f'Unknown {variant}')

def cfgs(suite,ds):
 if suite=='baseline':
  if ds in ('inews','gne'):
   return [('majority','majority'),('tfidf','lr'),('tfidf','svm'),('tfidf','rf'),('stats','rf'),('minilm','lr'),('minilm','mlp'),('llm','lr'),('llm','rf'),('tf_llm','rf'),('mini_llm','rf'),('full_text','lr')]
  if ds=='mirage':
   return [('majority','majority'),('tfidf','lr'),('minilm','lr'),('cliptext','lr'),('clipimage','lr'),('dinov2','lr'),('similarity','lr'),('clipconcat','lr'),('clipconcat','rf'),('dino_text','lr'),('clip_llm_text','rf'),('clip_llm','rf'),('late','lr'),('late_llm','rf')]
 if suite=='ablation':
  if ds in ('inews','gne'):return [(x,'lr') for x in ['full_text','no_stats','no_llm','no_mini','no_uncertainty','no_urgency','no_frame','no_stance']]
  if ds=='mirage':return [(x,'lr') for x in ['late_llm','no_cliptext','no_clipimage','no_dino','no_sem','no_visual','no_text','no_similarity','shuffled_image','clipconcat']]
 if suite=='vlm' and ds=='mirage':return [(x,'lr') for x in ['clipconcat','clip_llm','vlm','vlm_llm']]
 return []

def make_estimator(kind,seed,reg=False):
 from sklearn.linear_model import LogisticRegression,Ridge
 from sklearn.svm import LinearSVC
 from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
 from sklearn.neural_network import MLPClassifier
 from sklearn.dummy import DummyClassifier
 if reg:
  return Ridge(alpha=10.)
 if kind=='majority':return DummyClassifier(strategy='most_frequent')
 if kind=='lr':return LogisticRegression(C=1,max_iter=400,class_weight='balanced',random_state=seed)
 if kind=='svm':return LinearSVC(C=1,class_weight='balanced',random_state=seed)
 if kind=='rf':return RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',n_jobs=8,random_state=seed,class_weight='balanced_subsample')
 if kind=='mlp':return MLPClassifier(hidden_layer_sizes=(128,),early_stopping=True,max_iter=90,random_state=seed)
 raise ValueError(kind)
def probabilities(model,X,classes):
 from scipy.special import softmax
 if hasattr(model,'predict_proba'):
  probs=model.predict_proba(X);ref=model.classes_
 else:
  raw=model.decision_function(X)
  if raw.ndim==1:raw=np.column_stack([-raw,raw])
  probs=softmax(raw,axis=1);ref=model.classes_
 out=np.zeros((X.shape[0],len(classes)),np.float32)
 for k,c in enumerate(ref):
  if c in classes:out[:,classes.index(c)]=probs[:,k]
 z=out.sum(axis=1,keepdims=True);return np.divide(out,z,out=np.zeros_like(out),where=z>0)
def scores(y,p,classes):
 from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
 predicted=np.asarray(classes)[np.asarray(p).argmax(axis=1)]
 return {'accuracy':float(accuracy_score(y,predicted)),'macro_f1':float(f1_score(y,predicted,labels=classes,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,predicted))}
def predict_model(ds,variant,kind,seed=42,source=None,regression=None,subset=None):
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.preprocessing import StandardScaler
 from sklearn.pipeline import make_pipeline
 from scipy.sparse import hstack,csr_matrix
 df=read_df(ds);b=blocks(ds)
 if subset=='vlm':
  cache=ROOT/'cache'/'llm'/f'{ds}_vlm.jsonl'
  if not cache.exists():raise FileNotFoundError(f'VLM cache missing: {cache}')
  ids=set()
  for line in cache.open(encoding='utf-8'):
   try:rec=json.loads(line);ids.add(str(rec['uid']))
   except Exception:pass
  keep=df.uid.astype(str).isin(ids).to_numpy()
  df=df.loc[keep].reset_index(drop=True);b={k:np.asarray(v)[keep] for k,v in b.items()}
 if regression is not None:
  good=np.isfinite(pd.to_numeric(df[regression],errors='coerce').to_numpy())
  df=df.loc[good].reset_index(drop=True);b={k:np.asarray(v)[good] for k,v in b.items()}
 splits=df.split.astype(str).to_numpy();train=splits=='train';val=splits=='val';test=np.isin(splits,['test','test1','test2','test3','test4','test5'])
 if source is not None:
  sources=df.source.astype(str).to_numpy();train=train & (sources!=source);test=(splits=='train') & (sources==source)
  # Validation is never from held-out source
  val=val & (sources!=source)
 if variant=='tfidf' or variant=='tf_llm':
  tok=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,sublinear_tf=True)
  xtrain=tok.fit_transform(df.text[train].astype(str));xval=tok.transform(df.text[val].astype(str));xtest=tok.transform(df.text[test].astype(str))
  if variant=='tf_llm':
   sem=sem_features(ds);xtrain=hstack([xtrain,csr_matrix(sem[train])]).tocsr();xval=hstack([xval,csr_matrix(sem[val])]).tocsr();xtest=hstack([xtest,csr_matrix(sem[test])]).tocsr()
 else:
  matrix=selector(ds,variant,b,seed)
  if isinstance(matrix,str):raise ValueError(matrix)
  xtrain=matrix[train];xval=matrix[val];xtest=matrix[test]
  # Scaling based on TRAIN only; preserve feature ordering
  if kind in ('lr','svm','mlp') or regression is not None:
   scaler=StandardScaler(with_mean=False);xtrain=scaler.fit_transform(xtrain);xval=scaler.transform(xval);xtest=scaler.transform(xtest)
 if not train.any() or not val.any() or not test.any():raise ValueError(f'{ds} empty train/val/test, source={source}')
 y=df[regression].to_numpy(dtype=np.float32) if regression is not None else df.label.astype(str).to_numpy(dtype='U256')
 if regression is not None:
  model=make_estimator(kind,seed,reg=True);model.fit(xtrain,y[train]);pval=model.predict(xval);ptest=model.predict(xtest)
  from sklearn.metrics import mean_absolute_error,mean_squared_error
  met={'mae':float(mean_absolute_error(y[test],ptest)),'rmse':float(np.sqrt(mean_squared_error(y[test],ptest)))}
  from scipy.stats import pearsonr,spearmanr
  if len(ptest)>=3 and np.std(ptest)>1.e-8 and np.std(y[test])>1.e-8:
   met.update(pearson=float(pearsonr(y[test],ptest).statistic),spearman=float(spearmanr(y[test],ptest).statistic))
  return met,{'val_index':np.flatnonzero(val),'test_index':np.flatnonzero(test),'val_pred':pval,'test_pred':ptest,'val_y':y[val],'test_y':y[test]}
 classes=sorted(set(y[train])|set(y[val])|set(y[test]));model=make_estimator(kind,seed);model.fit(xtrain,y[train]);pv=probabilities(model,xval,classes);pt=probabilities(model,xtest,classes)
 if variant=='tfidf' and kind=='lr' and source is None and regression is None:
  import joblib
  modelpath=ROOT/'cache'/'trained'/f'{ds}_tfidf_lr_s{seed}.joblib';modelpath.parent.mkdir(parents=True,exist_ok=True)
  joblib.dump({'vectorizer':tok,'model':model,'classes':classes},modelpath)
 met=scores(y[test],pt,classes);met.update({'val_macro_f1':scores(y[val],pv,classes)['macro_f1'],'worst_class_recall':float(min(((pt.argmax(1)==classes.index(c)) & (y[test]==c)).sum()/max(1,(y[test]==c).sum()) for c in classes if (y[test]==c).any()))})
 if 'source' in df:
  fs=[]
  for s in np.unique(df.source[test].astype(str)):
   mask=(df.source[test].astype(str).to_numpy()==s)
   if mask.sum()>=20:fs.append(scores(y[test][mask],pt[mask],classes)['macro_f1'])
  if len(fs)>1:met['source_f1_gap']=float(max(fs)-min(fs));met['worst_source_f1']=float(min(fs))
 return met,{'val_index':np.flatnonzero(val),'test_index':np.flatnonzero(test),'val_pred':pv,'test_pred':pt,'val_y':y[val],'test_y':y[test],'classes':np.array(classes,dtype='U100')}

def save_prediction(ds,tag,seed,result,cache=True):
 if not cache:return
 p=ROOT/'cache'/'predictions'/ds/f'{tag}_s{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True)
 clean={k:(np.asarray(v,dtype='U256') if isinstance(v,np.ndarray) and v.dtype.kind=='O' else v) for k,v in result.items()}
 np.savez_compressed(p,**clean)
def suite_runner(suite,seeds=SEEDS,force=False):
 data=['inews','gne','mirage'] if suite in ('baseline','ablation') else ['mirage'] if suite=='vlm' else ['inews','gne']
 out=ROOT/'results'/f'{suite}.csv';out.parent.mkdir(parents=True,exist_ok=True)
 completed=set()
 if out.exists() and not force:
  prior=pd.read_csv(out)
  prior=prior[prior.status=='ok']
  completed=set(zip(prior.dataset,prior.variant,prior.model,prior.seed))
 for ds in data:
  for k,(variant,kind) in enumerate(cfgs(suite,ds)):
   for seed in seeds:
    ident=(ds,f'{variant}_{k}',kind,seed)
    if ident in completed:continue
    log('RUN',suite,ds,ident)
    try:
     met,pred=predict_model(ds,variant,kind,seed,subset='vlm' if suite=='vlm' else None)
     save_prediction(ds,f'{suite}_{variant}_{kind}_{k}',seed,pred)
     row=dict(suite=suite,dataset=ds,variant=f'{variant}_{k}',feature=variant,model=kind,seed=seed,**met,status='ok')
    except Exception as e:
     log('FAIL',ds,variant,kind,repr(e));row=dict(suite=suite,dataset=ds,variant=f'{variant}_{k}',feature=variant,model=kind,seed=seed,status='failed',error=repr(e))
    append_table(out,row)
    completed.add(ident)
 log('SUITE DONE',suite,out)

def vad_runner(seeds=SEEDS):
 out=ROOT/'results'/'vad.csv';out.parent.mkdir(parents=True,exist_ok=True)
 done=set()
 if out.exists():
  a=pd.read_csv(out);a=a[a.status=='ok'];done=set(zip(a.feature,a.target,a.seed))
 for target in ('V','A','D'):
  for feat in ('tfidf','minilm','llm','full_text'):
   for seed in seeds:
    if (feat,target,seed) in done:continue
    log('VAD',feat,target,seed)
    try:
     met,_=predict_model('inews',feat,'ridge',seed,regression=target)
     row={'feature':feat,'target':target,'seed':seed,'status':'ok',**met}
    except Exception as ex:row={'feature':feat,'target':target,'seed':seed,'status':'failed','error':repr(ex)};log('ERROR',ex)
    append_table(out,row)

def intensity_runner(seeds=SEEDS):
 df=read_df('gne');labels=df.intensity.astype(str).str.lower().map({'low':0,'medium':1,'high':2})
 if labels.notna().sum()<100:log('INTENSITY unsupported labels',df.intensity.value_counts().to_dict());return
 # Here ordinal intensity is treated as regression; uses only data with annotated labels
 df=df.copy();df['intensity_score']=labels;save_df('_gne_intensity',df)
 # Features aligned by row; same ordering as GNE. Do not use dataset-specific feature path.
 for kind in ('stats','minilm'):
  import shutil
  src=featpath('gne',kind);dst=featpath('_gne_intensity',kind)
  if src.exists() and not dst.exists():dst.symlink_to(src)
  
 out=ROOT/'results'/'intensity.csv';out.parent.mkdir(parents=True,exist_ok=True)
 done=set()
 if out.exists():
  a=pd.read_csv(out);a=a[a.status=='ok'];done=set(zip(a.feature,a.seed))
 for feat in ('tfidf','minilm','llm','full_text'):
  for seed in seeds:
   if (feat,seed) in done:continue
   log('INTENSITY',feat,seed)
   # Regression uses GNE rows, pseudo label for ordinal intensity; sem and stats are aligned.
   try:
    b=blocks('gne');f=read_df('gne');keep=labels.notna().to_numpy();tr=(f.split=='train').to_numpy()&keep;te=(f.split=='test').to_numpy()&keep
    if feat=='tfidf':
     from sklearn.feature_extraction.text import TfidfVectorizer
     tx=TfidfVectorizer(min_df=2,max_features=12000).fit(f.text[tr].astype(str));X=tx.transform(f.text[tr].astype(str));Xt=tx.transform(f.text[te].astype(str))
    else:
     mat=selector('gne',feat,b,seed);X=mat[tr];Xt=mat[te]
    from sklearn.linear_model import Ridge
    mod=Ridge(alpha=10).fit(X,labels[tr]);pr=np.clip(mod.predict(Xt),0,2);target=labels[te].to_numpy()
    row={'feature':feat,'seed':seed,'status':'ok','mae':float(np.abs(pr-target).mean()),'rounded_accuracy':float((np.rint(pr)==target).mean())}
   except Exception as ex:row={'feature':feat,'seed':seed,'status':'failed','error':repr(ex)};log('ERROR',ex)
   append_table(out,row)

def source_runner(seeds=SEEDS):
 out=ROOT/'results'/'source_holdout.csv';out.parent.mkdir(parents=True,exist_ok=True);done=set()
 if out.exists():
  a=pd.read_csv(out);a=a[a.status=='ok'];done=set(zip(a.dataset,a.source,a.feature,a.seed))
 for ds in ('inews','gne'):
  df=read_df(ds);base=df[df.split.isin(['train','val'])]
  sources=base.source.value_counts();selected=[s for s,n in sources.items() if n>=40][:4]
  for source in selected:
   for feat in ('tfidf','mini_llm','full_text'):
    for seed in seeds:
     if (ds,source,feat,seed) in done:continue
     log('SOURCE',ds,source,feat,seed)
     try:
      met,_=predict_model(ds,feat,'lr',seed,source=source)
      row=dict(dataset=ds,source=source,feature=feat,seed=seed,status='ok',**met)
     except Exception as ex:row=dict(dataset=ds,source=source,feature=feat,seed=seed,status='failed',error=repr(ex));log('ERROR',ex)
     append_table(out,row)

def pareto(seeds=(42,43,44),pop=24,gens=10):
 # Multi-objective fusion across cached baseline VAL probabilities, never training on TEST.
 from sklearn.metrics import f1_score
 out=ROOT/'results'/'pareto.csv';out.parent.mkdir(parents=True,exist_ok=True)
 done=set()
 if out.exists():
  a=pd.read_csv(out);a=a[a.status=='ok'];done=set(zip(a.dataset,a.method,a.seed))
 for ds in ('inews','gne','mirage'):
  for seed in seeds:
   files=sorted((ROOT/'cache'/'predictions'/ds).glob(f'baseline_*_s{seed}.npz'))
   picked=[];packs=[]
   for p in files:
    try:
     dat=np.load(p)
     if 'classes' not in dat:continue
     classes=dat['classes'].tolist()
     if len(classes)<2:continue
     if packs and (list(packs[0]['classes'])!=classes or not np.array_equal(dat['val_index'],packs[0]['val_index']) or not np.array_equal(dat['test_index'],packs[0]['test_index'])):continue
     packs.append({k:dat[k] for k in dat.files});picked.append(p.stem)
    except Exception as ex:log('SKIP',p,ex)
   if len(packs)<3:log('PARETO missing baseline predictions',ds,seed);continue
   val=np.stack([p['val_pred'] for p in packs]);test=np.stack([p['test_pred'] for p in packs]);yval=packs[0]['val_y'];ytest=packs[0]['test_y'];classes=packs[0]['classes'].tolist();frame=read_df(ds);src=frame.iloc[packs[0]['val_index']].source.to_numpy();m=len(packs)
   def activate(w):
    a=np.asarray(w).copy();a[a<=.02]=0.
    if not a.any():a[np.argmax(w)]=1.
    return a/a.sum()
   def evaluate(w):
    a=activate(w);p=np.einsum('m,mnc->nc',a,val);pred=np.array(classes)[p.argmax(1)]
    mf=float(f1_score(yval,pred,labels=classes,average='macro',zero_division=0))
    rec=[]
    for c in classes:
     mask=yval==c
     if mask.sum():rec.append(float((pred[mask]==c).mean()))
    worst=min(rec) if rec else 0
    gap=0.
    groups=[s for s in np.unique(src) if (src==s).sum()>=5]
    if len(groups)>1:
     vals=[f1_score(yval[src==s],pred[src==s],labels=classes,average='macro',zero_division=0) for s in groups]
     gap=float(max(vals)-min(vals))
    return np.asarray([-mf,-worst,gap,float((activate(w)>0).sum())/m],float)
   def dominates(a,b):return bool(np.all(a<=b) and np.any(a<b))
   def nondominated(Q):
    F=np.asarray([evaluate(q) for q in Q]);idx=[i for i in range(len(Q)) if not any(dominates(F[j],F[i]) for j in range(len(Q)) if j!=i)];return Q[idx],F[idx]
   for alg in ('nsga2','random','single'):
    if (ds,alg,seed) in done:continue
    rng=np.random.default_rng(seed+{'nsga2':0,'random':1000,'single':2000}[alg]);Q=rng.dirichlet(np.ones(m),size=pop);budget=pop+gens*pop
    # Equal objective evaluation budget across all search methods
    if alg=='random':Q=rng.dirichlet(np.ones(m),size=budget)
    else:
     for g in range(gens):
      F=np.asarray([evaluate(w) for w in Q]);offs=[]
      for _ in range(pop):
       a,b=rng.choice(len(Q),2,replace=False)
       if alg=='single':parent=Q[a] if F[a,0]<F[b,0] else Q[b]
       else:parent=Q[a] if dominates(F[a],F[b]) else Q[b] if dominates(F[b],F[a]) else Q[rng.choice([a,b])]
       mutant=np.maximum(0.,parent+rng.normal(0,.09,size=m));mutant*=rng.random(m)>.1
       if mutant.sum()==0:mutant[rng.integers(m)]=1.
       offs.append(mutant/mutant.sum())
      joint=np.vstack([Q,offs])
      if alg=='single':Q=joint[np.argsort([evaluate(w)[0] for w in joint])[:pop]]
      else:
       # Lightweight NSGA-II: fast non-dominated sorting + crowding distance
       F=np.asarray([evaluate(w) for w in joint]);N=len(joint);S=[[] for _ in range(N)];n=np.zeros(N,int);fronts=[[]]
       for i in range(N):
        for j in range(N):
         if i==j:continue
         if dominates(F[i],F[j]):S[i].append(j)
         elif dominates(F[j],F[i]):n[i]+=1
        if not n[i]:fronts[0].append(i)
       k=0
       while k<len(fronts) and fronts[k]:
        nxt=[]
        for i in fronts[k]:
         for j in S[i]:
          n[j]-=1
          if n[j]==0:nxt.append(j)
        if nxt:fronts.append(nxt)
        k+=1
       keep=[]
       for front in fronts:
        if len(keep)+len(front)<=pop:keep.extend(front);continue
        crowd=np.zeros(len(front));fr=F[front]
        for col in range(fr.shape[1]):
         order=np.argsort(fr[:,col]);crowd[order[0]]=crowd[order[-1]]=np.inf
         span=max(1.e-12,fr[order[-1],col]-fr[order[0],col])
         for idx in range(1,len(front)-1):crowd[order[idx]]+=(fr[order[idx+1],col]-fr[order[idx-1],col])/span
        keep.extend([front[i] for i in np.argsort(-crowd)[:pop-len(keep)] ]);break
       Q=joint[keep]
    bests,ff=nondominated(Q);# predeclared validation rule: highest val macro-F1; ties: fewer active features
    choice=sorted(range(len(bests)),key=lambda i:(ff[i,0],ff[i,3],ff[i,2]))[0];w=bests[choice];weights=activate(w)
    pred=np.einsum('m,mnc->nc',weights,test);met=scores(ytest,pred,classes)
    frontdir=ROOT/'results'/'pareto_fronts';frontdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(ff,columns=['neg_val_f1','neg_worst_recall','source_f1_gap','fraction_active']).to_csv(frontdir/f'{ds}_{alg}_{seed}.csv',index=False)
    jsonsave(ROOT/'results'/'pareto_weights'/f'{ds}_{alg}_{seed}.json',{'candidates':picked,'weights':weights.tolist(),'val_objectives':ff[choice].tolist(),'test_metrics':met,'evaluations_budget':budget})
    # Fixed-reference 4D dominated hypervolume, Monte Carlo estimate.
    hv_rng=np.random.default_rng(2026);lower=np.array([-1.,-1.,0.,0.]);upper=np.array([0.,0.,1.,1.])
    pts=hv_rng.uniform(lower,upper,size=(4000,4));hypervolume=float(np.any(np.all(ff[:,None,:]<=pts[None,:,:],axis=2),axis=0).mean())
    row=dict(dataset=ds,method=alg,seed=seed,candidates=m,eval_budget=budget,non_dominated=len(bests),val_hypervolume_est=hypervolume,val_macro_f1=-ff[choice,0],active_features=int((weights>0).sum()),status='ok',**met)
    append_table(out,row);done.add((ds,alg,seed));log('PARETO',ds,alg,seed,met)

def robustness(seeds=SEEDS):
 """60 cached TF-IDF evaluations and 25 Mirage OOD domain evaluations."""
 import joblib
 from sklearn.metrics import f1_score
 out=ROOT/'results'/'robustness.csv';out.parent.mkdir(parents=True,exist_ok=True)
 data=[]
 def perturb(s,mode):
  if mode=='lowercase':return s.lower()
  if mode=='whitespace':return re.sub(r'\s+',' ',s).strip()
  if mode=='punctuation':return re.sub(r'(?<!\d)[,.!?;:](?!\d)',' ',s)
  if mode=='unicode':return s.replace('“','"').replace('”','"').replace('’',"'")
  raise ValueError(mode)
 for ds in ('inews','gne','mirage'):
  f=read_df(ds);mask=f.split.isin(['test','test1','test2','test3','test4','test5'])
  y=f.loc[mask,'label'].astype(str).to_numpy()
  for seed in seeds:
   cache=ROOT/'cache'/'trained'/f'{ds}_tfidf_lr_s{seed}.joblib'
   if not cache.exists():log('ROBUST missing trained',cache);continue
   saved=joblib.load(cache);tok=saved['vectorizer'];model=saved['model'];cl=saved['classes']
   for mode in ('lowercase','whitespace','punctuation','unicode'):
    textvals=f.loc[mask,'text'].astype(str).map(lambda s:perturb(s,mode))
    probs=probabilities(model,tok.transform(textvals),cl)
    data.append(dict(dataset=ds,seed=seed,method='tfidf_lr',perturbation=mode,n=len(y),**scores(y,probs,cl)))
  if ds=='mirage':
   for seed in seeds:
    path=ROOT/'cache'/'predictions'/ds/f'baseline_clip_llm_rf_11_s{seed}.npz'
    # Fallback to CLIP concatenation, which does not require an LLM
    if not path.exists():path=ROOT/'cache'/'predictions'/ds/f'baseline_clipconcat_lr_7_s{seed}.npz'
    if not path.exists():continue
    pred=np.load(path);testframe=f.iloc[pred['test_index']].reset_index(drop=True)
    for group in ['test1','test2','test3','test4','test5']:
     m=(testframe.split==group).to_numpy()
     if m.sum()==0:continue
     data.append(dict(dataset=ds,seed=seed,method=path.stem,perturbation='OOD_'+group,n=int(m.sum()),**scores(pred['test_y'][m],pred['test_pred'][m],pred['classes'].tolist())))
 if data:pd.DataFrame(data).to_csv(out,index=False);log('ROBUST',len(data),out)

def semeval_zero(modelkey='qwen2_5_7b_instruct',prompt_id=0,limit=0):
 """Independent human emotion intensity check; never train on SemEval test."""
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 f=read_df('semeval');out=ROOT/'cache'/'zeroshot'/f'semeval_{modelkey}_p{prompt_id}.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
 seen={}
 if out.exists():
  for line in out.open():
   try:d=json.loads(line);seen[d['uid']]=d
   except Exception:pass
 path=str(ROOT/'models'/modelkey);tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype='auto',device_map='auto',local_files_only=True).eval()
 out.parent.mkdir(parents=True,exist_ok=True)
 for j,row in enumerate(f.itertuples(index=False),1):
  if limit and j>limit:break
  if str(row.uid) in seen:continue
  prompt=('Estimate the emotional content expressed by the headline, not how you personally feel. '
   if prompt_id==0 else 'Act as a conservative emotion-intensity annotator. Assign intensity only if explicitly justified by the headline. ')
  prompt+='Return JSON ONLY: anger, disgust, fear, joy, sadness, surprise each 0 to 100, valence -100 to 100. Headline: '+row.text
  msgs=[{'role':'user','content':prompt}]
  template_kw={'enable_thinking':False} if 'qwen3' in modelkey else {}
  inputs=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt',**template_kw).to(model.device)
  with torch.inference_mode():output=model.generate(inputs,max_new_tokens=135,do_sample=False,pad_token_id=tok.eos_token_id)
  obj=parse_json(tok.decode(output[0][inputs.shape[1]:],skip_special_tokens=True))
  if isinstance(obj,dict):
   record={'uid':str(row.uid),'scores':obj}
   with out.open('a') as h:h.write(json.dumps(record)+'\n')
  if j%100==0:log('SEMEVAL',modelkey,prompt_id,j,'/',len(f))
 log('SEMEVAL inference finished',modelkey,prompt_id)

def semeval_report():
 from scipy.stats import spearmanr,pearsonr
 df=read_df('semeval');rows=[]
 base=ROOT/'cache'/'zeroshot'
 for p in sorted(base.glob('semeval_*.jsonl')):
  pred={}
  for line in p.open():
   try:r=json.loads(line);pred[str(r['uid'])]=r['scores']
   except Exception:pass
  for key in [*EMOTIONS,'valence']:
   aa=[];bb=[]
   for r in df.itertuples(index=False):
    if str(r.uid) not in pred:continue
    try:v=float(pred[str(r.uid)].get(key))
    except Exception:continue
    aa.append(float(getattr(r,key)));bb.append(float(np.clip(v,-100 if key=='valence' else 0,100)))
   if len(aa)<3:continue
   rows.append(dict(dataset='SemEval',model=p.stem,emotion=key,n=len(aa),mae=float(np.abs(np.array(aa)-np.array(bb)).mean()),spearman=float(spearmanr(aa,bb).statistic),pearson=float(pearsonr(aa,bb).statistic)))
 if rows:
  out=ROOT/'results'/'semeval_zeroshot.csv';out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False);log('SEMEVAL REPORT',out)

def stress_controls():
 """Label-free visual pairing stress and SemEval inter-annotator variation."""
 from sklearn.preprocessing import normalize
 rows=[];rng=np.random.default_rng(42)
 try:
  df=read_df('clips');t=normalize(features('clips','cliptext'));v=normalize(features('clips','clipimage'))
  matched=(t*v).sum(axis=1)
  splits=df.source.astype(str).to_numpy()
  variants={
   'random_image':rng.permutation(len(df)),
   'cyclic_shift':np.roll(np.arange(len(df)),1),
   'reverse_order':np.arange(len(df))[::-1],
   'within_source':np.arange(len(df)),
  }
  for src in np.unique(splits):
   idx=np.flatnonzero(splits==src)
   if len(idx)>1:variants['within_source'][idx]=rng.permutation(idx)
  for name,index in variants.items():
   swapped=(t*v[index]).sum(axis=1)
   rows.append(dict(dataset='NewsCLIPpings',control=name,n=len(df),mean_original_similarity=float(matched.mean()),mean_shuffled_similarity=float(swapped.mean()),fraction_original_higher=float((matched>swapped).mean()),note='counterfactual image swapping changes target; no preserved-label accuracy is claimed'))
 except Exception as ex:log('STRESS clip failed',repr(ex))
 try:
  base=ROOT/'datasets'/'semeval2007_affective_text'/'Annotations';scores={}
  for p in sorted(base.glob('*.txt')):
   rec={}
   for ln in p.read_text(errors='replace').splitlines():
    v=ln.split()
    if len(v)<8:continue
    try:rec[int(v[0])]=[float(z) for z in v[1:8]]
    except ValueError:pass
   scores[p.stem]=rec
  raters=list(scores)
  for name,inds in [('six_emotions',list(range(6))),('valence',[6])]:
   diffs=[]
   for a in range(len(raters)):
    for b in range(a+1,len(raters)):
     common=set(scores[raters[a]])&set(scores[raters[b]])
     for k in common:
      va=np.array(scores[raters[a]][k]);vb=np.array(scores[raters[b]][k]);diffs.extend(np.abs(va[inds]-vb[inds]).tolist())
   if diffs:rows.append(dict(dataset='SemEval',control='rater_disagreement_'+name,n=len(diffs),mean_abs_rater_difference=float(np.mean(diffs)),note='pairwise inter-annotator absolute difference, not model error'))
 except Exception as ex:log('STRESS SemEval failed',repr(ex))
 if rows:
  p=ROOT/'results'/'stress_controls.csv';p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False);log('STRESS',p,len(rows))

def external():
 """NewsCLIPpings zero-shot image-caption compatibility + GNE evidence."""
 from sklearn.metrics import roc_auc_score,average_precision_score
 from sklearn.preprocessing import normalize
 out=ROOT/'results'/'external.csv';out.parent.mkdir(parents=True,exist_ok=True);rows=[]
 try:
  df=read_df('clips');truth=df.label.astype(int).to_numpy();t=normalize(features('clips','cliptext'));v=normalize(features('clips','clipimage'))
  raw=(t*v).sum(axis=1);sc={'CLIP_B32_cosine':raw}
  # Exploratory transductive normalizations use only unlabeled image/text pools.
  # They MUST NOT be described as non-transductive deployment baselines.
  sim=(t@v.T).astype(np.float32)
  im_mean=sim.mean(axis=0);tx_mean=sim.mean(axis=1)
  sc['B32_global_centered']=raw-im_mean-tx_mean
  for k in (5,10,20):
   rk=min(k,max(1,len(df)-1));left=np.partition(sim,-rk,axis=1)[:,-rk:].mean(1)
   right=np.partition(sim,-rk,axis=0)[-rk:,:].mean(0)
   sc[f'B32_CSLS_k{rk}']=2*raw-left-right
  # Rank-normalized alignment: percentile against all other captions and images.
  row_rank=(sim<=raw[:,None]).mean(1);col_rank=(sim<=raw[None,:]).mean(0)
  sc['B32_bidirectional_rank']=(row_rank+col_rank)/2
  del sim
  if featpath('clips','cliptextL').exists() and featpath('clips','clipimageL').exists():
   tl=normalize(features('clips','cliptextL'));vl=normalize(features('clips','clipimageL'))
   large=(tl*vl).sum(1);sc['CLIP_L14_cosine']=large
   sc['B32_L14_ensemble']=(raw+large)/2
   sc['B32_L14_centered']=((raw-np.mean(raw))/(np.std(raw)+1e-6)+(large-np.mean(large))/(np.std(large)+1e-6))/2
  for method,score in sc.items():
   rows.append(dict(dataset='NewsCLIPpings',task='out_of_context',method=method,n=len(df),auc=float(roc_auc_score(truth,-score)),ap=float(average_precision_score(truth,-score)),note='test-only zero-shot; higher mismatch score = -compatibility; transductive if CSLS/centered/rank'))
  # Matched 1k subset independent of label, allows Qwen-VL comparison.
  p=ROOT/'cache'/'llm'/'clips_vlm.jsonl';data={}
  if p.exists():
   for line in p.open():
    try:r=json.loads(line);data[r['uid']]=r['features']
    except Exception:pass
  idx=np.flatnonzero(df.uid.astype(str).isin(set(data)).to_numpy())
  if len(idx)>=50:
   for method,score in sc.items():
    rows.append(dict(dataset='NewsCLIPpings_VLM_matched',task='out_of_context',method=method,n=len(idx),auc=float(roc_auc_score(truth[idx],-score[idx])),ap=float(average_precision_score(truth[idx],-score[idx])),note='matched VLM subset, precommitted by hash'))
   vlm_score=np.array([float(data.get(str(df.iloc[j].uid),{}).get('image_text_alignment',.5)) for j in idx])
   rows.append(dict(dataset='NewsCLIPpings_VLM_matched',task='out_of_context',method='Qwen2.5_VL_7B_alignment',n=len(idx),auc=float(roc_auc_score(truth[idx],-vlm_score)),ap=float(average_precision_score(truth[idx],-vlm_score)),note='LLM alignment is an unsupervised scalar, not a trained falsification detector'))
 except Exception as e:log('EXTERNAL CLIPS failed',repr(e))
 try:
  df=read_df('gne');p=ROOT/'cache'/'llm'/'gne_llm.jsonl';mapping={}
  for line in p.open():
   r=json.loads(line);mapping[r['uid']]=r['features']
  f1s=[]
  for r in df.itertuples(index=False):
   if str(r.uid) not in mapping:continue
   try:human=json.loads(r.cue)
   except Exception:human=[]
   gold=set(re.findall(r'\w+',' '.join(' '.join(x) if isinstance(x,list) else str(x) for x in human).lower()))
   pred=set(re.findall(r'\w+',str(mapping[str(r.uid)].get('evidence','')).lower()))
   if not gold:continue
   pr=len(gold&pred)/max(1,len(pred));rec=len(gold&pred)/max(1,len(gold));f1s.append(2*pr*rec/max(1.e-9,pr+rec))
  if f1s:rows.append(dict(dataset='GNE',task='cue_overlap',method='Qwen2.5-7B',n=len(f1s),token_set_f1=float(np.mean(f1s)),note='token-set overlap, not span-level F1'))
 except Exception as e:log('EXTERNAL GNE evidence',repr(e))
 if rows:pd.DataFrame(rows).to_csv(out,index=False);log('EXTERNAL',len(rows),out)

def bootstrap():
 from sklearn.metrics import f1_score
 rng=np.random.default_rng(42);output=[]
 for ds in ('inews','gne','mirage'):
  for predfile in sorted((ROOT/'cache'/'predictions'/ds).glob('baseline_*_s42.npz')):
   d=np.load(predfile);cl=d['classes'].tolist();y=d['test_y'];p=d['test_pred'];pred=np.asarray(cl)[p.argmax(1)]
   df=read_df(ds);ids=df.iloc[d['test_index']].group.to_numpy();groups={g:np.where(ids==g)[0] for g in np.unique(ids)};keys=list(groups)
   vals=[]
   for _ in range(400):
    sample=rng.choice(keys,len(keys),replace=True);idx=np.concatenate([groups[g] for g in sample]);vals.append(f1_score(y[idx],pred[idx],labels=cl,average='macro',zero_division=0))
   output.append(dict(dataset=ds,model=predfile.stem,macro_f1=float(f1_score(y,pred,labels=cl,average='macro',zero_division=0)),ci_low=float(np.quantile(vals,.025)),ci_high=float(np.quantile(vals,.975))))
 if output:
  p=ROOT/'results'/'bootstrap_ci.csv';pd.DataFrame(output).to_csv(p,index=False);log('BOOTSTRAP',p)

def statistical_comparisons():
 """Paired grouped bootstrap for baseline vs proposed fusion, BH corrected p-values."""
 from sklearn.metrics import f1_score
 rng=np.random.default_rng(173);rows=[]
 for ds in ('inews','gne','mirage'):
  frame=read_df(ds)
  proposed='baseline_full_text_lr_11' if ds!='mirage' else 'baseline_late_llm_rf_13'
  control='baseline_tfidf_lr_1'
  for seed in SEEDS:
   a=ROOT/'cache'/'predictions'/ds/f'{proposed}_s{seed}.npz'
   b=ROOT/'cache'/'predictions'/ds/f'{control}_s{seed}.npz'
   if not a.exists() or not b.exists():continue
   x=np.load(a);y=np.load(b)
   if not np.array_equal(x['test_index'],y['test_index']):continue
   truth=x['test_y'];classes=x['classes'].tolist();pa=np.asarray(classes)[x['test_pred'].argmax(1)];pb=np.asarray(classes)[y['test_pred'].argmax(1)]
   groups=frame.iloc[x['test_index']].group.astype(str).to_numpy();uniq=np.unique(groups);members={g:np.where(groups==g)[0] for g in uniq};diffs=[]
   for _ in range(1000):
    sample=rng.choice(uniq,len(uniq),replace=True);idx=np.concatenate([members[q] for q in sample]);diffs.append(f1_score(truth[idx],pa[idx],average='macro',labels=classes,zero_division=0)-f1_score(truth[idx],pb[idx],average='macro',labels=classes,zero_division=0))
   low,high=np.quantile(diffs,[.025,.975]);pval=2*min((np.asarray(diffs)<=0).mean(),(np.asarray(diffs)>=0).mean());rows.append(dict(dataset=ds,seed=seed,proposed=proposed,control=control,delta_macro_f1=float(np.mean(diffs)),ci95_low=float(low),ci95_high=float(high),bootstrap_p_two_sided=float(min(1.,pval))))
 if rows:
  df=pd.DataFrame(rows);p=df.bootstrap_p_two_sided.to_numpy();order=np.argsort(p);adjusted=np.empty(len(p));v=1.
  for rank in range(len(p)-1,-1,-1):
   j=order[rank];v=min(v,p[j]*len(p)/(rank+1));adjusted[j]=v
  df['bh_q']=adjusted;out=ROOT/'results'/'paired_bootstrap_comparisons.csv';df.to_csv(out,index=False);log('STAT TESTS',out)

def figures():
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 except ImportError:log('Matplotlib unavailable: figures skipped');return
 fdir=ROOT/'results'/'figures';fdir.mkdir(parents=True,exist_ok=True)
 for table,metric in [('baseline','macro_f1'),('ablation','macro_f1'),('pareto','macro_f1')]:
  p=ROOT/'results'/f'{table}.csv'
  if not p.exists():continue
  df=pd.read_csv(p)
  if 'status' in df:df=df[df.status=='ok']
  if metric not in df or df.empty:continue
  field='feature' if 'feature' in df else 'method';agg=df.groupby(['dataset',field])[metric].mean().reset_index()
  for ds,sub in agg.groupby('dataset'):
   sub=sub.sort_values(metric);fig,ax=plt.subplots(figsize=(9,max(4,.33*len(sub))))
   ax.barh(sub[field],sub[metric]);ax.set_xlabel(metric.replace('_',' '));ax.set_xlim(0,1);ax.set_title(f'{table}: {ds}');fig.tight_layout()
   fig.savefig(fdir/f'{table}_{ds}.pdf');plt.close(fig)
 for p in sorted((ROOT/'results'/'pareto_fronts').glob('*.csv')):
  a=pd.read_csv(p)
  if len(a)<2:continue
  fig,ax=plt.subplots(figsize=(7,5));ax.scatter(-a.neg_val_f1,-a.neg_worst_recall,s=20)
  ax.set_xlabel('Validation macro-F1');ax.set_ylabel('Worst-class recall');ax.set_title(p.stem);fig.tight_layout();fig.savefig(fdir/f'{p.stem}.pdf');plt.close(fig)
 log('FIGURES',fdir)

def report():
 files=list((ROOT/'results').glob('*.csv'));rows=[]
 for p in files:
  try:
   a=pd.read_csv(p)
   if 'status' in a:a=a[a.status=='ok']
   rows.append({'file':p.name,'rows':len(a),'cols':','.join(a.columns),'status':'present'})
  except Exception as e:rows.append({'file':p.name,'status':'invalid','error':str(e)})
 out=ROOT/'reports'/'phase2_progress.json';jsonsave(out,{'tables':rows,'timestamp':time.strftime('%F %T')});log('REPORT',out)
 figures()
 for r in rows:log(r['file'],r.get('rows','ERROR'))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','extract','llm','baseline','ablation','vad','intensity','source','vlm','pareto','robustness','semeval','semeval_report','external','stress','bootstrap','statistics','report']);parser.add_argument('--dataset',default='mirage');parser.add_argument('--encoder',default='stats',choices=['stats','minilm','cliptext','clipimage','cliptextL','clipimageL','dinov2','llm','vlm']);parser.add_argument('--model',default='qwen2_5_7b_instruct');parser.add_argument('--prompt-id',type=int,default=0);parser.add_argument('--batch',type=int,default=12);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--force',action='store_true');parser.add_argument('--seeds',default='42,43,44,45,46');args=parser.parse_args();seeds=[int(x) for x in args.seeds.split(',')]
 random.seed(42);np.random.seed(42)
 if args.stage=='prepare':prepare(args.limit,args.force)
 elif args.stage=='extract':extract_stats(args.dataset) if args.encoder=='stats' else dense_extract(args.dataset,args.encoder,args.batch)
 elif args.stage=='llm':llm_extract(args.dataset,args.encoder,args.limit)
 elif args.stage in ('baseline','ablation','vlm'):suite_runner(args.stage,seeds,args.force)
 elif args.stage=='vad':vad_runner(seeds)
 elif args.stage=='intensity':intensity_runner(seeds)
 elif args.stage=='source':source_runner(seeds)
 elif args.stage=='pareto':pareto(seeds[:3])
 elif args.stage=='robustness':robustness(seeds)
 elif args.stage=='semeval':semeval_zero(args.model,args.prompt_id,args.limit)
 elif args.stage=='semeval_report':semeval_report()
 elif args.stage=='external':external()
 elif args.stage=='stress':stress_controls()
 elif args.stage=='bootstrap':bootstrap()
 elif args.stage=='statistics':statistical_comparisons()
 elif args.stage=='report':report()
if __name__=='__main__':main()
