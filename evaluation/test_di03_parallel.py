"""Model-free request identity and official-result-schema checks."""
import hashlib,importlib.util,json,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
spec=importlib.util.spec_from_file_location('wald_parallel',Path(__file__).with_name('di03_parallel.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def test_writes_each_identity_once_and_retains_unsupported_denominator(tmp_path,monkeypatch):
 rows=[{'state':s,'questions':{'q':{'type':'choice','criteria':{'a':'one','b':'two'}}},'_evaluation':{'run_id':str(i),'catalog_id':25,'payload_sha256':str(i)}} for i,s in enumerate(['normal','too long'])]
 class Suite:
  def __init__(self,*args):pass
  def verify(self,strict):assert strict;return {'edition':'0.3'}
  def rows(self):return iter(rows)
 class Engine:
  provenance={'kind':'test only'};latency='test only'
  def __init__(self,**opts):assert opts['model']=='04400-c18'
  def warmup(self):pass
  def __call__(self,state,questions):
   if state=='too long':raise M.Unsupported('maximum context length')
   return {'answers':{'q':{'type':'choice','choice':'a','probabilities':{'a':.8,'b':.2}}}},None
 monkeypatch.setattr(M,'Suite',Suite)
 boot=time.time();c={'status':'APPROVED_DI03_C16B_AUTO_FRESH_GPU_4H','model_id':'04400-c18','policy':'gate0.7','budget':512,'concurrency':64,'deadline_epoch':boot+14400,'GPU_boot_epoch':boot,'hostname':socket.gethostname(),'runner_sha256':M.sha(M.__file__),'source_pins':{},'suite_dir':'test-only','out_dir':str(tmp_path/'run'),'native_path':'test-only','temperature_path':'test-only','temperature_sha256':'test-only','root_contract_sha256':'test-only'}
 terminal=M.run(c,Engine)
 results=[json.loads(s) for s in (tmp_path/'run/results.jsonl').read_text().splitlines()]
 assert len(results)==2 and {r['run_id'] for r in results}=={'0','1'}
 assert terminal['counts']=={'ok':1,'unsupported':1}
 assert terminal['completed']==2 and terminal['expected']==140620
 assert terminal['status']=='BOUND_OR_FAILURE_PARTIAL'
 assert all('payload' not in r and r['total_wall_ms']>=0 for r in results)
