#!/usr/bin/env python3
"""NewsEmo phase 2 foreground runner: audited, restartable, no installations."""
import csv, fcntl, gc, json, math, os, re, subprocess, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path('/home/tahiti/NewsEmo')
PY = Path('/home/tahiti/Malashin_Projects/.venv_a100/bin/python')
QPY = Path('/home/tahiti/IconoBench/.venv_qwen/bin/python')
ENGINE = ROOT/'scripts/phase2.py'
NUM = ('uncertainty','urgency','causal_explicitness','attribution_strength','quotation_density','sensational_language')
TXT = ('frame','stance','focus','evidence')
VN = ('visual_salience','people_presence','scene_uncertainty','image_text_alignment','image_text_complementarity','visual_emphasis')
VT = ('scene','objects','activity','visual_evidence')
EMO = ('anger','disgust','fear','joy','sadness','surprise','valence')


def say(*args):
    print(time.strftime('%F %T'), *args, flush=True)


def cmd(label, executable, *args):
    say('START', label)
    result = subprocess.run([str(executable), '-u', str(ENGINE), *map(str,args)], cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f'{label}: subprocess returned {result.returncode}')
    say('DONE', label)


def good(field, value, kind='llm'):
    numeric = NUM if kind=='llm' else VN if kind=='vlm' else EMO
    if field in numeric:
        if isinstance(value, bool): return False
        try: n=float(value)
        except (ValueError,TypeError): return False
        lo = -100 if kind=='semeval' and field=='valence' else 0
        hi = 100 if kind=='semeval' else 1
        return math.isfinite(n) and lo <= n <= hi
    return isinstance(value,str) and (field in ('evidence','visual_evidence') or bool(value.strip()))


def merge_cache(path,kind='llm'):
    fields = NUM+TXT if kind=='llm' else VN+VT if kind=='vlm' else EMO
    cache={}
    malformed=0
    if path.exists():
        for line in path.open(encoding='utf-8'):
            try:
                entry=json.loads(line); key=str(entry['uid']); values=entry['scores' if kind=='semeval' else 'features']
                if not isinstance(values,dict): continue
                dst=cache.setdefault(key,{})
                for f in fields:
                    if good(f,values.get(f),kind):
                        dst[f] = float(values[f]) if f in (NUM+VN+EMO) else values[f]
            except (KeyError,ValueError,TypeError): malformed+=1
    return cache,malformed


def missing(record,kind):
    fields = NUM+TXT if kind=='llm' else VN+VT if kind=='vlm' else EMO
    return [f for f in fields if not good(f,record.get(f),kind)]


def candidate_rows(dataset,kind):
    import pandas as pd
    from hashlib import sha1
    df=pd.read_csv(ROOT/'cache/manifests'/f'{dataset}.csv',keep_default_na=False)
    if kind=='vlm':
        quotas={'train':2000,'val':500,**{f'test{i}':100 for i in range(1,6)}} if dataset=='mirage' else {'external':1000}
        parts=[]
        for split,quantity in quotas.items():
            a=df[(df.split==split)&df.image.astype(str).map(lambda p:bool(p) and Path(p).is_file())].copy()
            a=a.sort_values('uid',key=lambda s:s.map(lambda x:sha1(str(x).encode()).hexdigest()[:16])).head(quantity)
            if len(a)!=quantity:
                raise RuntimeError(f'{dataset} VLM split {split}: need {quantity} images, found {len(a)}')
            parts.append(a)
        df=pd.concat(parts,ignore_index=True)
    if not len(df) or df.uid.astype(str).duplicated().any():
        raise RuntimeError(f'{dataset} {kind}: missing or duplicated UIDs')
    return df


def parse_object(s):
    s=str(s).strip();begin=s.find('{')
    if begin<0:return {}
    for candidate in (s[begin:],s[begin:].replace("\\'", "'")):
        candidate=re.sub(r',\s*([}\]])',r'\1',candidate)
        try:
            obj,_=json.JSONDecoder(strict=False).raw_decode(candidate)
            if isinstance(obj,dict):return obj
        except (ValueError,TypeError):pass
    return {}


def extract_numbers(text,keys,kind):
    obj=parse_object(text);result={}
    for key in keys:
        value=obj.get(key)
        if good(key,value,kind):result[key]=float(value);continue
        m=re.search(r'(?<![\w])'+re.escape(key)+r'\s*["\']?\s*[:=]\s*["\']?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',text,re.I)
        if m and good(key,m.group(1),kind):result[key]=float(m.group(1))
    if len(keys)==1 and keys[0] not in result:
        m=re.fullmatch(r'\s*(?:```\w*\s*)?(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:```)?\s*',text)
        if m and good(keys[0],m.group(1),kind):result[keys[0]]=float(m.group(1))
    return result


def extract_strings(text,keys,kind):
    obj=parse_object(text);out={}
    for key in keys:
        if good(key,obj.get(key),kind):out[key]=obj[key];continue
        line=re.search(r'(?:^|\n)\s*["\']?'+re.escape(key)+r'["\']?\s*[:=]\s*(.*)',text,re.I)
        if line:
            val=line.group(1).strip().strip(', ').strip('"')
            if good(key,val,kind):out[key]=val
    if len(keys)==1 and keys[0] not in out and len(text.strip())<200:
        val=text.strip().strip('`').strip('"').strip()
        if good(keys[0],val,kind):out[keys[0]]=val
    return out


def model_session(kind,model_name=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,AutoProcessor
    if not torch.cuda.is_available():raise RuntimeError('GPU/CUDA unavailable: refusing CPU Qwen run')
    if kind=='vlm':
        from transformers import Qwen2_5_VLForConditionalGeneration
        path=ROOT/'models/qwen2_5_vl_7b'
        proc=AutoProcessor.from_pretrained(str(path),local_files_only=True)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path),torch_dtype='auto',device_map='auto',local_files_only=True).eval()
        return model,proc
    path=ROOT/'models'/(model_name or 'qwen2_5_7b_instruct')
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(path),torch_dtype='auto',device_map='auto',local_files_only=True).eval()
    return model,tok


def generate(model,proc,kind,prompt,imgpath=None,max_tokens=340,disable_thinking=False):
    import torch
    with torch.inference_mode():
        if kind=='vlm':
            from PIL import Image
            with Image.open(imgpath) as im:
                im=im.convert('RGB'); im.thumbnail((768,768)); im=im.copy()
            parts=[{'type':'image','image':im},{'type':'text','text':prompt}]
            rendered=proc.apply_chat_template([{'role':'user','content':parts}],tokenize=False,add_generation_prompt=True)
            inputs=proc(text=[rendered],images=[im],return_tensors='pt',padding=True).to(model.device)
            ids=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,temperature=None,top_p=None,top_k=None)
            answer=proc.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        else:
            kw={'enable_thinking':False} if disable_thinking else {}
            rendered=proc.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,**kw)
            inp=proc(rendered,return_tensors='pt',add_special_tokens=False,truncation=True,max_length=3200)
            inp={k:v.to(model.device) for k,v in inp.items()}
            ids=model.generate(**inp,max_new_tokens=max_tokens,do_sample=False,pad_token_id=proc.eos_token_id,temperature=None,top_p=None,top_k=None)
            answer=proc.decode(ids[0][inp['input_ids'].shape[1]:],skip_special_tokens=True)
    return answer.strip()


def prompt_full(text,kind):
    if kind=='llm':
        return ('Extract observable WRITING STYLE, not emotional reaction, sentiment class, or true/fake status. '
                'Return one COMPACT valid JSON object, no Markdown, all six keys with numbers from 0 to 1: '
                +', '.join(NUM)+'. Four short string keys: '+', '.join(TXT)
                +'. For evidence quote an exact short substring of the input, or empty string. No other keys. '
                'Each string maximum six words. TEXT: '+text[:2100])
    return ('Describe the image and supplied news caption without predicting true/fake or emotion targets. '
            'Return one COMPACT JSON object; numeric keys 0 to 1: '+', '.join(VN)
            +'. Short string keys: '+', '.join(VT)
            +'. visual_evidence is a short visual observation. CAPTION: '+text[:480])


def prompt_numeric(text,keys,kind):
    definitions={
      'uncertainty':'uncertain or hedged wording','urgency':'time urgency in wording',
      'causal_explicitness':'explicit causal statements','attribution_strength':'source attribution',
      'quotation_density':'direct quotations','sensational_language':'sensational wording',
      'visual_salience':'prominence of salient objects','people_presence':'visibility of people',
      'scene_uncertainty':'ambiguity of visible scene','image_text_alignment':'literal image-caption correspondence',
      'image_text_complementarity':'distinct extra information in image','visual_emphasis':'dramatic visual emphasis'}
    return ('Score only these observable dimensions: '+', '.join(f'{k} ({definitions[k]})' for k in keys)
            +'. Return ONLY key=number pairs separated by semicolons, numbers 0..1. '
            'Do not predict emotion or authenticity labels. CAPTION/TEXT: '+text[:1200])


def prompt_strings(text,keys,kind):
    return ('Return ONLY a compact valid JSON object with these string fields: '+', '.join(keys)
            +'. Each value maximum seven words. Evidence must be an exact quotation from text; visual_evidence must describe observed pixels. '
            'No emotion labels, no authenticity prediction. TEXT/CAPTION: '+text[:1200])


def cache_append(path,uid,features,kind):
    obj={'uid':uid,('scores' if kind=='semeval' else 'features'):features}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def semantic_stage(ds,kind):
    df=candidate_rows(ds,kind); path=ROOT/'cache/llm'/f'{ds}_{kind}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    existing,broken=merge_cache(path,kind)
    items=[(str(r.uid),str(r.text),str(getattr(r,'image',''))) for r in df.itertuples(index=False)]
    todo=[r for r in items if missing(existing.get(r[0],{}),kind)]
    say('AUDIT',ds,kind,'total',len(items),'complete',len(items)-len(todo),'remaining',len(todo),'invalid_jsonl_lines',broken)
    if not todo:return
    model,processor=model_session(kind)
    try:
        for i,(uid,text,imgpath) in enumerate(todo,1):
            state=dict(existing.get(uid,{}));need=missing(state,kind)
            # Incomplete records receive only missing fields; entirely absent ones get full extraction first.
            if len(need)>=9:
                try:
                    raw=generate(model,processor,kind,prompt_full(text,kind),imgpath,max_tokens=370)
                    state.update(extract_numbers(raw,[k for k in need if k in (NUM+VN)],kind))
                    state.update(extract_strings(raw,[k for k in need if k in (TXT+VT)],kind))
                except Exception as exc:say('FULL GENERATION ERROR',uid,repr(exc))
                if state:cache_append(path,uid,state,kind)
            for attempt in range(3):
                need=missing(state,kind)
                if not need:break
                numeric=[k for k in need if k in (NUM+VN)]
                strings=[k for k in need if k in (TXT+VT)]
                try:
                    if numeric:
                        raw=generate(model,processor,kind,prompt_numeric(text,numeric,kind),imgpath,max_tokens=120)
                        state.update(extract_numbers(raw,numeric,kind))
                    if strings:
                        raw=generate(model,processor,kind,prompt_strings(text,strings,kind),imgpath,max_tokens=160)
                        state.update(extract_strings(raw,strings,kind))
                    if state:cache_append(path,uid,state,kind)
                except Exception as exc:say('REPAIR ERROR',ds,kind,uid,attempt+1,repr(exc))
                if missing(state,kind) and attempt==1:
                    # Scalar fallbacks reduce schema failures.
                    for key in missing(state,kind):
                        try:
                            if key in (NUM+VN):
                                p=prompt_numeric(text,[key],kind)+' Return ONLY ONE NUMBER 0..1.'
                                raw=generate(model,processor,kind,p,imgpath,max_tokens=32)
                                state.update(extract_numbers(raw,[key],kind))
                            else:
                                raw=generate(model,processor,kind,prompt_strings(text,[key],kind),imgpath,max_tokens=64)
                                state.update(extract_strings(raw,[key],kind))
                        except Exception as exc:say('SCALAR ERROR',uid,key,repr(exc))
                    if state:cache_append(path,uid,state,kind)
            if missing(state,kind):say('INCOMPLETE',ds,kind,uid,missing(state,kind))
            if i%50==0 or i==len(todo):say('PROGRESS',ds,kind,i,'/',len(todo))
    finally:
        del model,processor;gc.collect()
        import torch
        torch.cuda.empty_cache()
    final,broken=merge_cache(path,kind)
    incomplete={uid:missing(final.get(uid,{}),kind) for uid,_,_ in items if missing(final.get(uid,{}),kind)}
    say('AUDIT FINAL',ds,kind,'complete',len(items)-len(incomplete),'/',len(items),'remaining',len(incomplete))
    if incomplete:
        (ROOT/'reports').mkdir(exist_ok=True)
        (ROOT/'reports'/f'phase2_missing_{ds}_{kind}.json').write_text(json.dumps(incomplete,indent=2))
        raise RuntimeError(f'{ds}/{kind} has {len(incomplete)} incomplete records; safe to rerun same command')


def semeval_validate(model_name,prompt_id):
    df=candidate_rows('semeval','llm');uids=list(df.uid.astype(str))
    path=ROOT/'cache/zeroshot'/f'semeval_{model_name}_p{prompt_id}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    old,broken=merge_cache(path,'semeval');todo=[(str(r.uid),str(r.text)) for r in df.itertuples(index=False) if missing(old.get(str(r.uid),{}),'semeval')]
    say('SEMEVAL AUDIT',model_name,prompt_id,'valid',len(uids)-len(todo),'/',len(uids))
    if todo:
        model,tok=model_session('semeval',model_name)
        try:
            for i,(uid,text) in enumerate(todo,1):
                state=old.get(uid,{}).copy()
                need=missing(state,'semeval')
                prompt=('Assess emotional content explicitly conveyed by this headline (not your personal feelings). '
                        'Return exactly one compact JSON object with numbers: anger, disgust, fear, joy, sadness, surprise from 0..100, '
                        'valence from -100..100. No other text. HEADLINE: '+text[:1000])
                for attempt in range(3):
                    if not need:break
                    request=prompt if attempt==0 else ('Reply ONLY with numeric key=value pairs for: '+', '.join(need)+'. '
                        'Six emotion intensities 0..100, valence -100..100. HEADLINE: '+text[:900])
                    try:
                        raw=generate(model,tok,'semeval',request,max_tokens=160,disable_thinking='qwen3' in model_name)
                        state.update(extract_numbers(raw,need,'semeval'))
                        cache_append(path,uid,state,'semeval')
                        need=missing(state,'semeval')
                    except Exception as exc:say('SEMEVAL RETRY ERROR',uid,repr(exc))
                if need:say('SEMEVAL INCOMPLETE',uid,need)
                if i%100==0 or i==len(todo):say('SEMEVAL',model_name,prompt_id,i,'/',len(todo))
        finally:
            del model,tok;gc.collect()
            import torch
            torch.cuda.empty_cache()
    final,_=merge_cache(path,'semeval')
    unfinished=[uid for uid in uids if missing(final.get(uid,{}),'semeval')]
    if unfinished:raise RuntimeError(f'{model_name} prompt {prompt_id}: {len(unfinished)} missing SemEval predictions')
    say('SEMEVAL COMPLETE',model_name,prompt_id,len(uids))


def verify_table(name,minimum=1):
    import pandas as pd
    path=ROOT/'results'/f'{name}.csv'
    if not path.exists():raise RuntimeError(f'Missing results/{name}.csv')
    df=pd.read_csv(path)
    if 'status' in df:
        good_rows=df[df.status=='ok'];errors=df[df.status=='failed']
        say('RESULT',name,'ok',len(good_rows),'failed_rows',len(errors))
    else:
        good_rows=df;errors=df.iloc[0:0];say('RESULT',name,'rows',len(df))
    if len(good_rows)<minimum:raise RuntimeError(f'{name}: expected at least {minimum} successful records; got {len(good_rows)}')
    # Historic failures can be retained in append-only result CSV after successful retry;
    # only flag configurations without any successful result.
    if 'status' in df and len(errors):
        keys={'baseline':['dataset','variant','model','seed'],'ablation':['dataset','variant','model','seed'],
              'vlm':['dataset','variant','model','seed'],'vad':['feature','target','seed'],
              'intensity':['feature','seed'],'source_holdout':['dataset','source','feature','seed'],
              'pareto':['dataset','method','seed']}.get(name)
        if keys:
            successes=set(map(tuple,good_rows[keys].astype(str).to_numpy()))
            unresolved=[x for x in map(tuple,errors[keys].astype(str).to_numpy()) if x not in successes]
            if unresolved:raise RuntimeError(f'{name}: {len(unresolved)} failed configurations; sample {unresolved[:4]}')
    return len(good_rows)


def validate_dependencies():
    for exe in (PY,QPY):
        if not exe.is_file():raise RuntimeError(f'Existing Python missing: {exe}')
    for exe,code in [(PY,'import torch,numpy,pandas,scipy,sklearn,pyarrow,transformers,PIL'),
                     (QPY,'import torch,transformers,PIL;from transformers import Qwen2_5_VLForConditionalGeneration')]:
        p=subprocess.run([str(exe),'-c',code],check=False)
        if p.returncode:raise RuntimeError(f'Existing env failed imports: {exe}')
    for name in ('qwen2_5_7b_instruct','qwen3_8b','qwen2_5_vl_7b','minilm_l6_v2','clip_vit_b32','clip_vit_l14','dinov2_small'):
        if not (ROOT/'models'/name).is_dir():raise RuntimeError(f'Missing model link {name}')
    say('PREFLIGHT PASSED')


def main():
    os.chdir(ROOT)
    for sub in ('logs','cache','results','reports'):(ROOT/sub).mkdir(exist_ok=True)
    lock=(ROOT/'logs/phase2_autorun.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another Phase 2 autorun is active')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1'
    validate_dependencies()
    cmd('prepare',PY,'prepare')
    for ds in ('inews','gne','semeval','mirage','clips'):cmd(f'stats_{ds}',PY,'extract','--dataset',ds,'--encoder','stats')
    for ds in ('inews','gne','mirage'):cmd(f'minilm_{ds}',PY,'extract','--dataset',ds,'--encoder','minilm','--batch','24')
    for ds in ('mirage','clips'):
        for enc,bs in (('cliptext','32'),('clipimage','16')):cmd(f'{enc}_{ds}',PY,'extract','--dataset',ds,'--encoder',enc,'--batch',bs)
    cmd('dino_mirage',PY,'extract','--dataset','mirage','--encoder','dinov2','--batch','16')
    for enc in ('cliptextL','clipimageL'):cmd(f'{enc}_clips',PY,'extract','--dataset','clips','--encoder',enc,'--batch','8')
    # Validate that dense feature extraction progress has actually reached all rows.
    import pandas as pd, numpy as np
    checks=[('inews','minilm'),('gne','minilm'),('mirage','minilm'),('mirage','cliptext'),('mirage','clipimage'),('mirage','dinov2'),
            ('clips','cliptext'),('clips','clipimage'),('clips','cliptextL'),('clips','clipimageL')]
    for ds,k in checks:
        p=ROOT/'cache/features'/f'{ds}_{k}.npy';progress=p.with_suffix('.progress.json')
        n=len(candidate_rows(ds,'llm'))
        mat=np.load(p,mmap_mode='r');done=json.loads(progress.read_text())
        if mat.shape[0]!=n or int(done.get('next',0))!=n:raise RuntimeError(f'Incomplete dense extraction {ds}/{k}')
        # Scan chunkwise to avoid making a second copy of a large matrix.
        for a in range(0,n,512):
            if not np.isfinite(mat[a:a+512]).all():raise RuntimeError(f'NaN/inf features {ds}/{k}')
    say('DENSE FEATURES VERIFIED')
    for ds in ('inews','gne','mirage'):semantic_stage(ds,'llm')
    for ds in ('mirage','clips'):semantic_stage(ds,'vlm')
    for stage,filename,minimum in [
       ('baseline','baseline',190),('vad','vad',60),('intensity','intensity',20),
       ('ablation','ablation',130),('source','source_holdout',1),('vlm','vlm',20),
       ('pareto','pareto',27),('robustness','robustness',60),('external','external',11),
       ('stress','stress_controls',6)]:
        cmd(stage,PY,stage)
        if stage=='source':
            expected_sources=0
            for ds in ('inews','gne'):
                d=candidate_rows(ds,'llm')
                counts=d[d.split.isin(['train','val'])].source.value_counts()
                expected_sources+=min(4,sum(n>=40 for n in counts))*3*5
            minimum=expected_sources
            say('SOURCE HOLDOUT CONFIGURATIONS POSSIBLE',expected_sources,'of 120 planned')
        verify_table(filename,minimum)
    for name in ('qwen2_5_7b_instruct','qwen3_8b'):
        for p in (0,1):semeval_validate(name,p)
    for stage,filename,minimum in [('semeval_report','semeval_zeroshot',28),('bootstrap','bootstrap_ci',38),('statistics','paired_bootstrap_comparisons',15)]:
        cmd(stage,PY,stage);verify_table(filename,minimum)
    cmd('report',PY,'report')
    verify_table('baseline',190)
    say('PHASE 2 ALL STAGES COMPLETED — check reports/phase2_progress.json')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        say('INTERRUPTED; caches preserved. Relaunch same script to resume.');sys.exit(130)
    except Exception as ex:
        say('STOP',repr(ex),'— caches preserved; inspect log and rerun after correction');sys.exit(1)
