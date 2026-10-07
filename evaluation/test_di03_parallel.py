"""Model-free request identity and official-result-schema checks."""
import hashlib,importlib.util,json,socket,sys,time
import pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
spec=importlib.util.spec_from_file_location('wald_parallel',Path(__file__).with_name('di03_parallel.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

@pytest.mark.parametrize('until_complete', [False, True])
@pytest.mark.parametrize('replicas', [1,4])
def test_writes_each_identity_once_and_retains_unsupported_denominator(tmp_path,monkeypatch,until_complete,replicas):
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
 if until_complete:c.update(status='APPROVED_DI03_C16B_AUTO_UNTIL_COMPLETE',deadline_epoch=None,shutdown_on_completion=True)
 if replicas==4:c.update(replicas=4,replica_rank=0)
 terminal=M.run(c,Engine)
 results=[json.loads(s) for s in (tmp_path/'run/results.jsonl').read_text().splitlines()]
 assigned=[r for r in rows if replicas==1 or int(hashlib.sha256(r['_evaluation']['run_id'].encode()).hexdigest(),16)%replicas==0]
 assert len(results)==len(assigned) and {r['run_id'] for r in results}=={r['_evaluation']['run_id'] for r in assigned}
 assert terminal['completed']==len(assigned)
 if replicas==1:
  assert terminal['counts']=={'ok':1,'unsupported':1}
  assert terminal['expected']==140620 and terminal['status']=='BOUND_OR_FAILURE_PARTIAL'
 else:assert terminal['expected']==len(assigned) and terminal['status']=='ALL_PUBLIC_REQUESTS_ATTEMPTED'
 assert all('payload' not in r and r['total_wall_ms']>=0 for r in results)


def test_four_replica_partition_is_disjoint_and_complete():
 ids=[str(i) for i in range(1000)]
 parts=[{rid for rid in ids if int(hashlib.sha256(rid.encode()).hexdigest(),16)%4==rank} for rank in range(4)]
 assert set.union(*parts)==set(ids)
 assert sum(map(len,parts))==len(ids)
 assert all(not parts[a]&parts[b] for a in range(4) for b in range(a+1,4))
