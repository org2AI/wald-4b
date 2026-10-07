"""Bounded public DI inference, compatible with the official results schema.

One immutable request per run_id. No power actions or model loading. Declared
concurrent throughput measurements are not leaderboard serial latency. Score
with the unmodified, pinned official 0.3 kit after inference has ended.
"""
import argparse,collections,concurrent.futures as cf,hashlib,json,os,socket,sys,threading,time
from pathlib import Path
from decision_index.engines.base import Unsupported, validate
from decision_index.runner import stamp
from decision_index.suite.io import Suite,dumps
from di03_native_auto_engine import WaldNativeAuto

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(p,v):
 tmp=Path(str(p)+'.tmp')
 with tmp.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 tmp.replace(p)

def run(c,engine_factory=WaldNativeAuto):
 assert c['status']=='APPROVED_DI03_C16B_AUTO_FRESH_GPU_4H'
 assert c['model_id']=='04400-c18' and c['policy']=='gate0.7' and c['budget']==512
 assert c['concurrency'] in (64,128) and c['deadline_epoch']==c['GPU_boot_epoch']+14400
 assert socket.gethostname()==c['hostname']
 assert sha(__file__)==c['runner_sha256']
 assert time.time()<c['deadline_epoch']-180
 for p,h in c['source_pins'].items():assert sha(p)==h
 suite=Suite(c['suite_dir'],'0.3');verified=suite.verify(strict=True)
 assert verified['edition']=='0.3'
 out=Path(c['out_dir']);out.mkdir(exist_ok=False)
 opts=dict(native_path=c['native_path'],temperature_path=c['temperature_path'],temperature_sha256=c['temperature_sha256'],deadline_epoch=c['deadline_epoch'],endpoint='http://127.0.0.1:8371',model=c['model_id'],max_model_len=131072,reference_root=c.get('reference_root'))
 first=engine_factory(**opts)
 environment={'engine':'wald-v2-native-auto0.7','model_source':first.provenance,'engine_options':opts,'concurrency':c['concurrency'],'frozen_corpus':verified,'latency':first.latency,'runner_sha256':sha(__file__),'root_contract_sha256':c['root_contract_sha256'],'boot_epoch':c['GPU_boot_epoch'],'deadline_epoch':c['deadline_epoch'],'no_model_startup_or_power_operations':True}
 write_json(out/'environment.json',environment)
 # The first client runs only benchmark-free synthetic warmup.
 first.warmup();local=threading.local();seen=set();counts=collections.Counter();start=time.time();stop=False
 def one(row):
  e=row['_evaluation'];t=time.perf_counter();r={**e,'engine':'wald-v2-native-auto0.7','started_utc':stamp()}
  try:
   if time.time()>=c['deadline_epoch']-180:return None
   if not hasattr(local,'engine'):local.engine=engine_factory(**opts)
   response,_=local.engine(row['state'],row['questions']);validate(row['questions'],response)
   r.update(status='ok',response=response)
  except Unsupported as exc:r.update(status='unsupported',error=str(exc))
  except Exception as exc:r.update(status='error',exception=type(exc).__name__,error=str(exc))
  ms=(time.perf_counter()-t)*1000;r.update(completed_utc=stamp(),total_wall_ms=ms,model_request_wall_ms=ms);return r
 rows=iter(suite.rows());f=(out/'results.jsonl').open('x')
 try:
  with cf.ThreadPoolExecutor(max_workers=c['concurrency']) as pool:
   pending={}
   def submit():
    row=next(rows,None)
    if row is None:return False
    rid=row['_evaluation']['run_id'];assert rid not in seen;seen.add(rid)
    pending[pool.submit(one,row)]=rid;return True
   for _ in range(c['concurrency']):
    if not submit():break
   while pending:
    if time.time()>=c['deadline_epoch']-180:stop=True
    done,_=cf.wait(pending,timeout=1,return_when=cf.FIRST_COMPLETED)
    for future in done:
     pending.pop(future);r=future.result()
     if r is not None:
      f.write(dumps(r)+'\n');f.flush();counts[r['status']]+=1
      if r.get('exception') in ('OutOfMemoryError','AcceleratorError') or 'device-side assert' in r.get('error',''):stop=True
     if not stop:submit()
    if counts['error']>=5 and not counts['ok']:stop=True
    write_json(out/'status.json',{'status':'DRAINING_AT_BOUND_OR_FAILURE' if stop else 'EVALUATING','completed':sum(counts.values()),'expected':140620,'counts':dict(counts),'epoch':time.time(),'elapsed_seconds':time.time()-start,'deadline_epoch':c['deadline_epoch'],'official_serial_latency_measured':False})
 finally:f.flush();os.fsync(f.fileno());f.close()
 record={'status':'ALL_PUBLIC_REQUESTS_ATTEMPTED' if sum(counts.values())==140620 else 'BOUND_OR_FAILURE_PARTIAL','counts':dict(counts),'expected':140620,'completed':sum(counts.values()),'epoch':time.time(),'results_sha256':sha(out/'results.jsonl'),'score_pending':True,'actual_provider_OFF_pending':True,'no_policy_selection_from_TEST':True}
 write_json(out/'terminal.json',record)
 return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
 assert sha(a.contract)==a.sha256;c=json.loads(Path(a.contract).read_text());c['root_contract_sha256']=a.sha256
 print(json.dumps(run(c)))
if __name__=='__main__':main()
