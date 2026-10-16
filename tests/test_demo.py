"""HTTP behavior of the standalone demo, without downloading a model."""
import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
import wave

import numpy as np

from demo.app import MAX_TEXT, Server, Voice


class DemoHTTP(unittest.TestCase):
    def setUp(self):
        self.voice = Voice()
        self.voice.model = type('Model', (), {'sample_rate': 24000,
            '__call__': lambda self, text: np.zeros(2400, dtype=np.float32)})()
        self.voice.state = 'ready'
        self.server = Server(('127.0.0.1', 0), self.voice)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None, headers=None, raw=None):
        data = json.dumps(payload).encode() if payload is not None else raw
        request = urllib.request.Request(self.url + path, data=data,
            headers=headers if headers is not None else {'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    def test_real_wav_and_generation_metrics(self):
        status, headers, body = self.request('/api/synthesize', {'text': 'Bonjour.'})
        self.assertEqual(status, 200)
        result = json.loads(body)
        with wave.open(io.BytesIO(base64.b64decode(result['audio']))) as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()),
                             (1, 2, 24000, 2400))
        self.assertEqual(result['audio_seconds'], 0.1)
        self.assertGreater(result['generation_seconds'], 0)
        self.assertAlmostEqual(result['realtime_speed'] * result['generation_seconds'], 0.1)
        self.assertFalse(self.voice.generating.locked())

    def test_invalid_input_and_cross_origin_requests(self):
        for payload in ({'text': ''}, {'text': 42}, {'text': 'x'*(MAX_TEXT+1)}, [], {'other': 'text'}):
            with self.subTest(payload_type=type(payload).__name__):
                self.assertEqual(self.request('/api/synthesize', payload)[0], 400)
        self.assertEqual(self.request('/api/synthesize', raw=b'{invalid')[0], 400)
        self.assertEqual(self.request('/api/synthesize', raw=b'{}', headers={'Content-Type':'text/plain'})[0], 415)
        self.assertEqual(self.request('/api/synthesize', {'text':'Bonjour'},
            headers={'Content-Type':'application/json', 'Origin':'https://example.invalid'})[0], 403)
        self.assertEqual(self.request('/../pyproject.toml')[0], 404)

    def test_loading_busy_and_failed_generation(self):
        self.voice.state = 'loading'
        self.assertEqual(self.request('/api/synthesize', {'text':'Bonjour'})[0], 503)
        self.voice.state = 'ready'
        self.voice.generating.acquire()
        self.assertEqual(self.request('/api/synthesize', {'text':'Bonjour'})[0], 409)
        self.voice.generating.release()
        class FailingModel:
            def __call__(self, text):
                raise RuntimeError('Test failure')
        self.voice.model = FailingModel()
        self.assertEqual(self.request('/api/synthesize', {'text':'Bonjour'})[0], 500)
        self.assertFalse(self.voice.generating.locked())

    def test_page_and_status(self):
        status, headers, body = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers['Content-Type'])
        self.assertIn('Temps de génération', body.decode())
        status, _, body = self.request('/api/status')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['state'], 'ready')


if __name__ == '__main__':
    unittest.main()
