"""Public CPU API and integrity-checked, versioned model cache."""
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import urllib.request
import zipfile

from .audio import PostProcessing
from .onnx_runtime import OnnxTTS

__version__ = '0.1.4'


def _message(text, verbose):
    if verbose:
        print('[Paradis-TTS] ' + text, file=sys.stderr, flush=True)


def _progress_class(verbose):
    from tqdm.auto import tqdm

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(disable=not verbose, file=sys.stderr, ascii=True)
            super().__init__(*args, **kwargs)

    return Progress


def _verify(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Model checksum mismatch: {name}')
    return root


def download_model(variant='fp32', *, weight_kind=None, cache_dir=None, local_files_only=False, verbose=True):
    """Download pinned Hugging Face weights once; reuse the verified local cache."""
    if type(verbose) is not bool:
        raise TypeError('verbose must be a boolean')
    if variant not in ('fp32', 'int8'):
        raise ValueError('variant must be fp32 or int8')
    registry = json.loads(Path(__file__).with_name('release_assets.json').read_text(encoding='utf-8'))
    default_kind = registry.get('default_weight_kind')
    if weight_kind is not None and weight_kind not in ('ema', 'raw'):
        raise ValueError('weight_kind must be ema or raw')
    selected_kind = weight_kind or default_kind
    variants = registry['variants']
    if selected_kind != default_kind:
        alternatives = registry.get('alternative_weights', {})
        if selected_kind not in alternatives:
            raise ValueError(f'Weights unavailable in this release: {selected_kind}')
        variants = alternatives[selected_kind]
    asset = variants[variant]
    label = ' '.join(part.upper() for part in (selected_kind, variant) if part)
    base = Path(cache_dir) if cache_dir else Path(os.environ.get('PARADIS_TTS_CACHE',
        str(Path(os.environ.get('LOCALAPPDATA', Path.home() / '.cache')) / 'paradis-tts')))
    destination = base / registry['release']
    if selected_kind is not None:
        destination = destination / selected_kind
    destination = destination / variant
    if destination.is_dir():
        verified = _verify(destination)
        _message(f'Modèle trouvé dans le cache : {label}.', verbose)
        return verified
    if local_files_only:
        raise FileNotFoundError(f'Model absent from cache: {destination}. Load once online or pass model_folder=...')
    destination.parent.mkdir(parents=True, exist_ok=True)
    size_label = f' ({asset["bytes"]/1e6:.1f} Mo)' if asset.get('bytes') else ''
    _message(f'Téléchargement depuis Hugging Face{size_label} : {label}...', verbose)
    Progress = _progress_class(verbose)
    with tempfile.TemporaryDirectory(prefix='download-', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / 'model.zip'
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'Paradis-TTS/' + __version__})
        def store(source, progress=None):
            digest = hashlib.sha256()
            size = 0
            with archive_path.open('wb') as output:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > 100 * 1024 * 1024:
                        raise ValueError('Unexpectedly large model archive')
                    digest.update(block)
                    output.write(block)
                    if progress is not None:
                        progress.update(len(block))
            return size, digest
        size, digest = 0, hashlib.sha256()
        failed = False
        try:
            with Progress(total=asset.get('bytes'), unit='B', unit_scale=True,
                          desc='[Paradis-TTS] Téléchargement HF') as progress:
                with urllib.request.urlopen(request, timeout=60) as response:
                    size, digest = store(response, progress)
        except (OSError, HTTPException):
            failed = True
        if failed or size == 0 or size != asset.get('bytes', size):
            # The official client also supports Xet/CAS when the CDN closes a
            # direct HTTP transfer early. Keep independent SHA256 validation.
            from huggingface_hub import hf_hub_download
            _message('Transfert direct incomplet ; reprise avec le client Hugging Face...', verbose)
            downloaded = hf_hub_download(repo_id=registry['model_repository'],
                filename=asset['url'].rsplit('/',1)[1], revision=registry['model_revision'],
                cache_dir=temporary/'hub', token=False, tqdm_class=Progress)
            with Path(downloaded).open('rb') as source:
                size, digest = store(source)
        _message('Vérification SHA256...', verbose)
        if digest.hexdigest() != asset['sha256']:
            raise ValueError('Downloaded archive checksum mismatch')
        _message('Extraction et vérification des fichiers du modèle...', verbose)
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
    verified = _verify(destination)
    _message(f'Modèle enregistré dans le cache : {verified}', verbose)
    return verified


class ParadisTTS(OnnxTTS):
    """French mono-voice TTS. Returns a float32, mono, 24 kHz NumPy array.

    Postprocessing is enabled by default. Pass post_traitement=False for raw
    decoder audio, or postprocessing=PostProcessing(...) to customize it.
    Initialization and download messages go to stderr; verbose=False hides them.
    """
    sample_rate = 24000

    def __new__(cls, bundle=None, *args, **kwargs):
        if bundle is not None and kwargs.get('model_folder') is not None:
            raise ValueError('Pass model_folder or bundle, not both')
        # Keep explicitly supplied research bundles usable by existing notebooks.
        if bundle is not None:
            manifest = Path(bundle) / 'manifest.json'
            if manifest.is_file() and 'graphs' not in json.loads(manifest.read_text(encoding='utf-8')):
                verbose = kwargs.pop('verbose', True)
                if type(verbose) is not bool:
                    raise TypeError('verbose must be a boolean')
                from .inference import ParadisTTS as ResearchTTS
                _message('Chargement du modèle de recherche local...', verbose)
                model = ResearchTTS(bundle, *args, **kwargs)
                _message('Modèle prêt (24 kHz).', verbose)
                return model
        return super().__new__(cls)

    def __init__(self, bundle=None, *, model_folder=None, post_traitement=True, variant='fp32', weight_kind=None, threads=2,
                 cache_dir=None, local_files_only=False, postprocessing=None, verbose=True):
        """Load model_folder directly, or download variant from Hugging Face.

        model_folder must contain manifest.json and the extracted ONNX graphs.
        Explicit local paths never fall back to downloading. bundle is a legacy
        alias retained for existing applications and research notebooks.
        """
        if type(verbose) is not bool:
            raise TypeError('verbose must be a boolean')
        if model_folder is not None:
            bundle = model_folder
        if type(post_traitement) is not bool:
            raise TypeError('post_traitement must be a boolean')
        if not post_traitement and postprocessing is not None:
            raise ValueError('Custom postprocessing conflicts with post_traitement=False')
        if bundle is None:
            adjacent = Path(__file__).resolve().parent.parent
            bundle = adjacent if (adjacent / 'manifest.json').is_file() else download_model(
                variant, weight_kind=weight_kind, cache_dir=cache_dir, local_files_only=local_files_only, verbose=verbose)
        else:
            bundle = Path(bundle).expanduser().resolve()
            if not (bundle / 'manifest.json').is_file():
                raise FileNotFoundError(f'Local model folder must contain manifest.json: {bundle}')
            _message(f'Modèle local : {bundle}', verbose)
        if not post_traitement:
            postprocessing = PostProcessing(phase_lock_frames=0, presence_restore=False,
                dynamic_soften_db=0, upper_soften_db=0, attack_soften_db=0,
                syllable_soften_db=0, dynamics_smoothing=0, softening_match_level=False)
        _message('Chargement du modèle sur CPU...', verbose)
        super().__init__(bundle, threads=threads, postprocessing=postprocessing)
        self.post_traitement = post_traitement
        self.verbose = verbose
        _message('Modèle prêt (24 kHz).', verbose)
