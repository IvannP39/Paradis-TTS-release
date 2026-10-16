"""Public API switches, pinned downloads and hostile archive rejection."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import zipfile

from paradis_tts import ParadisTTS
from paradis_tts.api import download_model
from paradis_tts.audio import PostProcessing
from paradis_tts.onnx_runtime import OnnxTTS
from paradis_tts.normalization_v2 import VERSION


class PublicAPI(unittest.TestCase):
    def test_incomplete_direct_download_uses_verified_official_hf_fallback(self):
        content=b'graph'
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as stream:
            stream.writestr('decoder.onnx',content)
            stream.writestr('manifest.json',json.dumps({'files':{
                'decoder.onnx':hashlib.sha256(content).hexdigest()}}))
        blob=archive.getvalue()
        registry={'release':'fallback-test','model_repository':'owner/model','model_revision':'pinned-commit',
            'variants':{'fp32':{'url':'https://example.invalid/model.zip','bytes':len(blob),
                               'sha256':hashlib.sha256(blob).hexdigest()}}}
        original=Path.read_text
        def read(path,*args,**kwargs):
            return json.dumps(registry) if path.name=='release_assets.json' else original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'download.zip'
            source.write_bytes(blob)
            with patch.object(Path,'read_text',read), patch('urllib.request.urlopen',return_value=io.BytesIO(b'')), patch(
                    'huggingface_hub.hf_hub_download',return_value=str(source)) as fallback:
                directory=download_model(cache_dir=Path(temporary)/'cache')
                self.assertEqual((directory/'decoder.onnx').read_bytes(),content)
                self.assertEqual(fallback.call_args.kwargs['revision'],'pinned-commit')
                self.assertFalse(fallback.call_args.kwargs['token'])
            source.write_bytes(b'corrupt')
            with patch.object(Path,'read_text',read), patch('urllib.request.urlopen',return_value=io.BytesIO(b'')), patch(
                    'huggingface_hub.hf_hub_download',return_value=str(source)):
                with self.assertRaisesRegex(ValueError,'checksum'):
                    download_model(cache_dir=Path(temporary)/'different-cache')

    def test_old_model_profiles_cannot_reenable_dynamic_softening_by_default(self):
        previous=PostProcessing(dynamic_soften_db=1.)
        for settings_file in (False,True):
            with self.subTest(settings_file=settings_file), tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary)
                (folder/'manifest.json').write_text(json.dumps({'schema_version':1,
                    'normalizer_version':VERSION,'sample_rate':24000,'files':{},
                    'espeak_version':[1,2,3],'inference_defaults':previous.to_dict()}))
                (folder/'vocab.json').write_text('{}')
                if settings_file: previous.save(folder/'postprocessing.json')
                frontend=Mock()
                frontend.g2p.backend.version.return_value=[1,2,3]
                with patch('onnxruntime.InferenceSession'), patch(
                        'paradis_tts.onnx_runtime.InferenceFrontend',return_value=frontend):
                    model=OnnxTTS(folder)
                    self.assertEqual(model.postprocessing.dynamic_soften_db,0.)
                    self.assertEqual(model.postprocessing.phase_lock_frames,97)
                    self.assertTrue(model.postprocessing.presence_restore)
                    self.assertEqual(model.postprocessing.upper_soften_db,1.5)
                    explicit=OnnxTTS(folder,postprocessing=previous)
                    self.assertEqual(explicit.postprocessing,previous)

    def test_ema_default_and_raw_alternative_have_separate_offline_caches(self):
        blobs = {}
        for kind in ('ema', 'raw'):
            content = kind.encode()
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('decoder.onnx', content)
                stream.writestr('manifest.json', json.dumps({'files':{
                    'decoder.onnx':hashlib.sha256(content).hexdigest()}}))
            blobs[kind] = archive.getvalue()
        assets = {kind:{'fp32':{'url':'https://example.invalid/'+kind,
            'sha256':hashlib.sha256(blob).hexdigest()}} for kind,blob in blobs.items()}
        registry = {'release':'test-new-model', 'default_weight_kind':'ema',
                    'variants':assets['ema'], 'alternative_weights':{'raw':assets['raw']}}
        original = Path.read_text
        def read(path, *args, **kwargs):
            return json.dumps(registry) if path.name=='release_assets.json' else original(path,*args,**kwargs)
        def respond(request, **kwargs):
            return io.BytesIO(blobs[request.full_url.rsplit('/',1)[1]])
        with tempfile.TemporaryDirectory() as temporary, patch.object(Path,'read_text',read):
            with patch('urllib.request.urlopen',side_effect=respond) as network:
                ema = download_model(cache_dir=temporary)
                raw = download_model(weight_kind='raw',cache_dir=temporary)
                self.assertEqual(network.call_count,2)
            self.assertNotEqual(ema,raw)
            self.assertEqual((ema/'decoder.onnx').read_bytes(),b'ema')
            self.assertEqual((raw/'decoder.onnx').read_bytes(),b'raw')
            with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected network')):
                self.assertEqual(download_model(weight_kind='ema',cache_dir=temporary,local_files_only=True),ema)
                self.assertEqual(download_model(weight_kind='raw',cache_dir=temporary,local_files_only=True),raw)
            with self.assertRaises(ValueError):
                download_model(weight_kind='other',cache_dir=temporary)

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
