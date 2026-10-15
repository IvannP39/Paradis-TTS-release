"""Public CPU API and integrity-checked, versioned model cache."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import urllib.request
import zipfile

from .audio import PostProcessing
from .onnx_runtime import OnnxTTS

__version__ = '0.1.0'


def _verify(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Model checksum mismatch: {name}')
    return root


def download_model(variant='fp32', *, cache_dir=None, local_files_only=False):
    """Download once from the v0 release; subsequent loads work without a network."""
    if variant not in ('fp32', 'int8'):
        raise ValueError('variant must be fp32 or int8')
    registry = json.loads(Path(__file__).with_name('release_assets.json').read_text(encoding='utf-8'))
    asset = registry['variants'][variant]
    base = Path(cache_dir) if cache_dir else Path(os.environ.get('PARADIS_TTS_CACHE',
        str(Path(os.environ.get('LOCALAPPDATA', Path.home() / '.cache')) / 'paradis-tts')))
    destination = base / registry['release'] / variant
    if destination.is_dir():
        return _verify(destination)
    if local_files_only:
        raise FileNotFoundError(f'Model absent from cache: {destination}. Load once online or pass bundle=...')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='download-', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / 'model.zip'
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'Paradis-TTS/' + __version__})
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open('wb') as output:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 100 * 1024 * 1024:
                    raise ValueError('Unexpectedly large model archive')
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != asset['sha256']:
            raise ValueError('Downloaded archive checksum mismatch')
        extracted = temporary / 'model'
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            if sum(member.file_size for member in archive.infolist()) > 150 * 1024 * 1024:
                raise ValueError('Unexpectedly large extracted model')
            for member in archive.infolist():
                target = (extracted / member.filename).resolve()
                if not target.is_relative_to(extracted.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Invalid archive path')
            archive.extractall(extracted)
        _verify(extracted)
        try:
            extracted.rename(destination)
        except FileExistsError:
            _verify(destination)  # Another process completed the same download.
    return _verify(destination)


class ParadisTTS(OnnxTTS):
    """French mono-voice TTS. Returns a float32, mono, 24 kHz NumPy array.

    Postprocessing is enabled by default. Pass post_traitement=False for raw
    decoder audio, or postprocessing=PostProcessing(...) to customize it.
    """
    sample_rate = 24000

    def __new__(cls, bundle=None, *args, **kwargs):
        # Keep explicitly supplied research bundles usable by existing notebooks.
        if bundle is not None:
            manifest = Path(bundle) / 'manifest.json'
            if manifest.is_file() and 'graphs' not in json.loads(manifest.read_text(encoding='utf-8')):
                from .inference import ParadisTTS as ResearchTTS
                return ResearchTTS(bundle, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, bundle=None, *, post_traitement=True, variant='fp32', threads=2,
                 cache_dir=None, local_files_only=False, postprocessing=None):
        if type(post_traitement) is not bool:
            raise TypeError('post_traitement must be a boolean')
        if not post_traitement and postprocessing is not None:
            raise ValueError('Custom postprocessing conflicts with post_traitement=False')
        if bundle is None:
            adjacent = Path(__file__).resolve().parent.parent
            bundle = adjacent if (adjacent / 'manifest.json').is_file() else download_model(
                variant, cache_dir=cache_dir, local_files_only=local_files_only)
        if not post_traitement:
            postprocessing = PostProcessing(phase_lock_frames=0, presence_restore=False,
                dynamic_soften_db=0, upper_soften_db=0, attack_soften_db=0,
                syllable_soften_db=0, dynamics_smoothing=0, softening_match_level=False)
        super().__init__(bundle, threads=threads, postprocessing=postprocessing)
        self.post_traitement = post_traitement
