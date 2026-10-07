"""C16B native Auto 0.7 adapter for the official Decision Index Engine API.

Uses the previously evaluated frozen native reader, not the released legacy
plain/Reasoning server. Supplies only state/questions to inference. No launch,
training, upload or power operations. Parallel-run timings are not leaderboard
single-request latency measurements.
"""
import hashlib
import importlib.util
import time
from pathlib import Path
from decision_index.engines.base import Engine, Unsupported, validate

NATIVE_SHA = '764f3a683becd16b87b9ce9bfc01a5fe483ef26c9a3345a724b82780d73ea73a'
MODEL_SHA = '2859df99ed481592cdc3731fd768bf29ccd63273b2dab623642ad3b38b0e14c4'

class WaldNativeAuto(Engine):
    name = 'wald-v2-native-auto0.7'
    latency = 'Request wall time at declared concurrency; not official serial leaderboard latency.'

    def __init__(self, native_path, temperature_path, temperature_sha256,
                 deadline_epoch, endpoint='http://127.0.0.1:8371',
                 model='04400-c18', max_model_len=131072, **options):
        super().__init__(**options)
        assert model == '04400-c18', 'This contract binds C16B only'
        path = Path(native_path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == NATIVE_SHA
        temp = Path(temperature_path)
        assert hashlib.sha256(temp.read_bytes()).hexdigest() == temperature_sha256
        self.deadline_epoch = float(deadline_epoch)
        assert self.deadline_epoch > time.time()
        spec = importlib.util.spec_from_file_location('wald_frozen_native', path)
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)
        self.client, self.sov = self.native.make_client(
            endpoint, model, int(max_model_len),
            time.monotonic() + self.deadline_epoch - time.time())
        self.table = self.sov.load_tables(temp).get('A')
        self.model = model
        self.provenance = {'model': model, 'model_sha256': MODEL_SHA,
            'native_reader_sha256': NATIVE_SHA,
            'temperature_sha256': temperature_sha256,
            'policy': 'native Auto 0.7; untempered A gate; at most 512 thought tokens; original option order',
            'prompt_format': 'repeat_state_plain in native chat',
            'wide': 'frozen onepass knockout; every supplied option receives probability',
            'context_limit': int(max_model_len), 'no_truncation': True,
            'new_training': False}

    def __call__(self, state, questions):
        if time.time() >= self.deadline_epoch - 120:
            raise TimeoutError('Registered evaluation deadline reached')
        request = {'state': state, 'questions': questions}
        try:
            response = self.native.answer_task(self.client, self.sov,
                {'request': request, 'suite': 'di'}, self.table, 0.7, 512)
        except self.sov.Capacity as exc:
            raise Unsupported(str(exc)) from exc
        validate(questions, response)
        return response, response
