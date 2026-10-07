"""CPU-only checks of the public-kit adapter boundary; no model access."""
import importlib.util
from pathlib import Path
import pytest
from decision_index.engines.base import Unsupported

PATH=Path(__file__).with_name('di03_native_auto_engine.py')
spec=importlib.util.spec_from_file_location('wald_di03_adapter',PATH)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Capacity(ValueError): pass
class Sov: Capacity=Capacity

def engine(callback):
 e=mod.WaldNativeAuto.__new__(mod.WaldNativeAuto)
 e.deadline_epoch=mod.time.time()+1000;e.client=object();e.sov=Sov();e.table={'single':1.0}
 class Native: answer_task=staticmethod(callback)
 e.native=Native();return e

def test_auto_contract_sends_only_inference_fields_and_fixed_settings():
 def answer(client,sov,task,table,gate,budget):
  assert set(task)=={'request','suite'} and task['suite']=='di'
  assert set(task['request'])=={'state','questions'}
  assert gate==0.7 and budget==512
  return {'answers':{'q':{'type':'choice','choice':'a','probabilities':{'a':.6,'b':.4}}}}
 q={'q':{'type':'choice','instructions':'choose','criteria':{'a':'one','b':'two'}}}
 response,raw=engine(answer)('state',q)
 assert response is raw

def test_incomplete_option_mapping_is_rejected():
 def answer(*args):return {'answers':{'q':{'type':'choice','choice':'a','probabilities':{'a':1.0}}}}
 q={'q':{'type':'choice','criteria':{'a':'one','b':'two'}}}
 with pytest.raises(ValueError,match='Incomplete'):engine(answer)('state',q)

def test_declared_capacity_maps_to_official_unsupported():
 def answer(*args):raise Capacity('maximum context length')
 with pytest.raises(Unsupported):engine(answer)('state',{})

def test_budget_expiry_prevents_an_inference_call():
 def answer(*args):raise AssertionError('must not call model')
 e=engine(answer);e.deadline_epoch=mod.time.time()+119
 with pytest.raises(TimeoutError):e('state',{})


def test_until_complete_has_no_time_cutoff():
 def answer(*args):return {"answers":{"q":{"type":"choice","choice":"a","probabilities":{"a":.6,"b":.4}}}}
 e=engine(answer);e.deadline_epoch=None
 e("state",{"q":{"type":"choice","criteria":{"a":"one","b":"two"}}})
