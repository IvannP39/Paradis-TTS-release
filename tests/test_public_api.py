"""Public API switches, pinned downloads and hostile archive rejection."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from paradis_tts import ParadisTTS
from paradis_tts.api import download_model


class PublicAPI(unittest.TestCase):
    def test_explicit_local_folder_never_downloads_and_legacy_alias_works(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'manifest.json').write_text(json.dumps({'graphs':[]}))
            with patch('paradis_tts.api.download_model', side_effect=AssertionError('Unexpected download')), patch(
                    'paradis_tts.onnx_runtime.OnnxTTS.__init__', return_value=None) as init:
                ParadisTTS(model_folder=folder, local_files_only=True)
                self.assertEqual(init.call_args.args[0], folder.resolve())
                ParadisTTS(bundle=folder)
                with self.assertRaisesRegex(ValueError, 'not both'):
                    ParadisTTS(bundle=folder, model_folder=folder)
                with self.assertRaisesRegex(FileNotFoundError, 'manifest.json'):
                    ParadisTTS(model_folder=folder/'missing')

    def test_postprocessing_is_default_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle/'manifest.json').write_text(json.dumps({'graphs':[]}))
            with patch('paradis_tts.onnx_runtime.OnnxTTS.__init__', return_value=None) as init:
                model = ParadisTTS(bundle)
                self.assertTrue(model.post_traitement)
                self.assertIsNone(init.call_args.kwargs['postprocessing'])
                model = ParadisTTS(bundle, post_traitement=False)
                self.assertFalse(model.post_traitement)
                settings = init.call_args.kwargs['postprocessing']
                self.assertEqual(settings.phase_lock_frames, 0)
                self.assertFalse(settings.presence_restore)
                self.assertEqual(settings.dynamic_soften_db+settings.upper_soften_db, 0)
                with self.assertRaises(TypeError):
                    ParadisTTS(bundle, post_traitement='yes')

    def test_first_download_then_offline_cache_and_corruption(self):
        payload = b'graph'
        files = {'text.onnx':hashlib.sha256(payload).hexdigest()}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('text.onnx', payload)
            stream.writestr('manifest.json', json.dumps({'files':files}))
        blob = archive.getvalue()
        registry = {'release':'test-v0', 'variants':{'fp32':{
            'url':'https://example.invalid/model.zip', 'sha256':hashlib.sha256(blob).hexdigest()}}}
        original = Path.read_text

        def read(path, *args, **kwargs):
            return json.dumps(registry) if path.name=='release_assets.json' else original(path,*args,**kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(Path, 'read_text', read):
            with self.assertRaises(FileNotFoundError):
                download_model(cache_dir=temporary, local_files_only=True)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(blob)) as network:
                directory = download_model(cache_dir=temporary)
                network.assert_called_once()
            with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
                self.assertEqual(download_model(cache_dir=temporary, local_files_only=True),directory)
            (directory/'text.onnx').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                download_model(cache_dir=temporary)

    def test_bad_hash_and_zip_slip_are_rejected(self):
        for malicious_path, digest in (('safe', 'wrong'), ('../escape', None)):
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr(malicious_path, b'payload')
            blob = archive.getvalue()
            registry = {'release':'test', 'variants':{'fp32':{'url':'https://example.invalid/archive',
                'sha256':digest or hashlib.sha256(blob).hexdigest()}}}
            with tempfile.TemporaryDirectory() as temporary:
                with patch.object(Path,'read_text',return_value=json.dumps(registry)), patch(
                        'urllib.request.urlopen',return_value=io.BytesIO(blob)):
                    with self.assertRaises(ValueError):
                        download_model(cache_dir=temporary)
                self.assertFalse((Path(temporary)/'escape').exists())


if __name__ == '__main__':
    unittest.main()
