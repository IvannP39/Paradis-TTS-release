"""Resumable CPU synthesis, French Whisper WER and DNSMOS P.835 on a frozen held-out set."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def word_text(text):
    # Same normalization on reference and hypothesis: accents retained, apostrophes
    # and hyphens become word boundaries. Numerals expand with the runtime normalizer.
    from paradis_tts.normalization_v2 import normalize_sentence
    text = normalize_sentence(text)[0].casefold()
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text), flags=re.UNICODE))


def environment(threads):
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    return {'cpu':cpu, 'threads':threads, 'os':platform.platform(), 'python':platform.python_version(),
            'numpy':np.__version__, 'onnxruntime':version('onnxruntime'), 'device':'cpu',
            'concurrent_gpu_training':True, 'timing_note':'Shared laptop during independent GPU training; timings are not an isolated performance claim.'}


class Memory:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.poll, daemon=True)
        self.worker.start()

    def poll(self):
        while not self.stop.wait(.02):
            self.peak = max(self.peak, self.process.memory_info().rss)


def synthesize(args, config, rows):
    import soundfile as sf
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    directory = args.output/args.variant
    directory.mkdir(parents=True, exist_ok=True)
    contract = {'corpus_sha256':sha(args.corpus), 'variant':args.variant,
                'config_sha256':sha(args.config), 'environment':environment(args.threads)}
    if args.variant in ('fp32','int8'):
        contract['model_manifest_sha256'] = sha(args.bundle/args.variant/'manifest.json')
    if (directory/'contract.json').exists():
        previous = json.loads((directory/'contract.json').read_text(encoding='utf-8'))
        if previous != contract:
            raise ValueError('Synthesis contract changed; choose a fresh output')
    else:
        write(directory/'contract.json', contract)
    memory = Memory()
    loaded = time.perf_counter()
    if args.variant in ('fp32','int8'):
        from paradis_tts.onnx_runtime import OnnxTTS
        model = OnnxTTS(args.bundle/args.variant, threads=args.threads)
        rate = 24000
        def generate(text, seed):
            return model(text, seed=seed)
        details = {'sample_rate':rate, 'postprocessing':model.postprocessing.to_dict(),
                   'parameters':model.manifest['parameters'], 'normalization':'fr-runtime-2',
                   'precision':args.variant, 'integer_compute':False}
    elif args.variant == 'piper':
        from piper import PiperVoice
        from piper.config import PiperConfig, SynthesisConfig
        c = config['piper']
        path = hf_hub_download(c['repo'], c['file'], revision=c['revision'])
        config_path = hf_hub_download(c['repo'], c['file']+'.json', revision=c['revision'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        model = PiperVoice(config=PiperConfig.from_dict(json.loads(Path(config_path).read_text())),
            session=ort.InferenceSession(path, options, providers=['CPUExecutionProvider']))
        rate = model.config.sample_rate
        def generate(text, seed):
            return np.concatenate([chunk.audio_float_array for chunk in model.synthesize(text, SynthesisConfig())])
        details = {'sample_rate':rate, 'model_sha256':sha(path), 'bytes':Path(path).stat().st_size,
                   'repo':c, 'runtime':version('piper-tts'),
                   'seed_note':'Piper native ONNX random excitation is not controlled by the Paradis seed.'}
    else:
        import torch
        from kokoro import KModel, KPipeline
        torch.set_num_threads(args.threads)
        c = config['kokoro']
        paths = {name:hf_hub_download(c['repo'], name, revision=c['revision'])
                 for name in ('config.json','kokoro-v1_0.pth','voices/ff_siwis.pt')}
        model = KModel(config=paths['config.json'], model=paths['kokoro-v1_0.pth']).eval().to('cpu')
        pipeline = KPipeline(lang_code='f', model=model, device='cpu', repo_id=c['repo'])
        voice = torch.load(paths['voices/ff_siwis.pt'], weights_only=True, map_location='cpu')
        rate = 24000
        @torch.inference_mode()
        def generate(text, seed):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return np.concatenate([result.audio.detach().cpu().numpy() for result in pipeline(text, voice=voice, speed=1)])
        details = {'sample_rate':rate, 'repo':c, 'runtime':version('kokoro'),
                   'parameters':sum(p.numel() for p in model.parameters()),
                   'model_sha256':sha(paths['kokoro-v1_0.pth']), 'bytes':Path(paths['kokoro-v1_0.pth']).stat().st_size}
    load_seconds = time.perf_counter()-loaded
    generate('Bonjour tout le monde.', 1729)  # Excluded warm-up, includes G2P.
    log_path = directory/'synthesis.jsonl'
    previous = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()] if log_path.exists() else []
    done = {row['id'] for row in previous}
    with log_path.open('a', encoding='utf-8') as log:
        for index, row in enumerate(rows):
            if row['id'] in done:
                continue
            started = time.perf_counter()
            audio = np.asarray(generate(row['normalized_text'], row['seed']), dtype=np.float32).reshape(-1)
            elapsed = time.perf_counter()-started
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError(f'Invalid audio: {row["id"]}')
            path = directory/(row['id']+'.wav')
            sf.write(path, audio, rate, subtype='PCM_16')
            result = {'id':row['id'], 'seconds':len(audio)/rate, 'elapsed_seconds':elapsed,
                'rtf':elapsed/(len(audio)/rate), 'sample_rate':rate, 'samples':len(audio),
                'peak':float(np.max(np.abs(audio))), 'clipped_fraction':float(np.mean(np.abs(audio)>1)),
                'wav_sha256':sha(path)}
            log.write(json.dumps(result)+'\n')
            log.flush()
            if (index+1)%10 == 0:
                print(f'SYNTHESIS {args.variant} {index+1}/{len(rows)} RTF={result["rtf"]:.3f}', flush=True)
    memory.stop.set()
    write(directory/'synthesis_report.json', {'status':'COMPLETE', 'count':len(rows),
        'model':details, 'environment':contract['environment'], 'load_seconds':load_seconds,
        'peak_process_rss_bytes':memory.peak, 'warmup_excluded':True,
        'timing_includes':'Normalization, G2P, model and configured DSP; excludes load and WAV disk write'})


def score(args, config, rows):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    import jiwer
    import onnxruntime as ort
    import requests
    from scipy.signal import resample_poly
    import soundfile as sf
    c = config['asr']
    c = dict(c, device=args.asr_device, compute_type=args.asr_compute)
    if args.asr_device == 'cuda' and sys.platform == 'win32':
        # Reuse installed CUDA/cuDNN DLLs without adding Torch to the eval env.
        dll_path = ROOT/'.venv-cuda/Lib/site-packages/torch/lib'
        if dll_path.is_dir():
            dll_handle = os.add_dll_directory(str(dll_path))
            os.environ['PATH'] = str(dll_path)+os.pathsep+os.environ['PATH']
    path = snapshot_download(c['repo'], revision=c['revision'])
    asr = WhisperModel(path, device=c['device'], compute_type=c['compute_type'], cpu_threads=args.threads, num_workers=1)
    dns_path = args.output/'dnsmos.onnx'
    if not dns_path.exists():
        response = requests.get(config['dnsmos']['url'], timeout=60)
        response.raise_for_status()
        dns_path.write_bytes(response.content)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    dns = ort.InferenceSession(str(dns_path), options, providers=['CPUExecutionProvider'])
    coefficients = ([-.08397278,1.22083953,.0052439], [-.13166888,1.60915514,-.39604546],
                    [-.06766283,1.11546468,.04602535])

    def dns_score(audio):
        # Microsoft non-personalized P.835: repeat short clips; 9.01-second windows,
        # one-second stride; calibrate each segment then average. P.808 is not used.
        count = int(9.01*16000)
        signal = audio
        while len(signal) < count:
            signal = np.concatenate((signal, signal))
        scores = []
        for start in range(int(np.floor(len(signal)/16000)-9.01)+1):
            segment = signal[start*16000:start*16000+count]
            prediction = dns.run(None, {'input_1':segment[None].astype(np.float32)})[0][0]
            scores.append([float(np.polyval(p, v)) for p,v in zip(coefficients,prediction)])
        values = np.mean(scores, axis=0)
        return dict(zip(('sig','bak','ovrl'), map(float,values)))

    for variant in args.variants:
        directory = args.output/variant
        if not args.follow:
            synth = json.loads((directory/'synthesis_report.json').read_text(encoding='utf-8'))
            if synth['count'] != len(rows):
                raise ValueError('Synthesis is incomplete')
        contract = {'corpus_sha256':sha(args.corpus), 'config_sha256':sha(args.config),
                    'dnsmos_sha256':sha(dns_path), 'asr':c, 'faster_whisper':version('faster-whisper'),
                    'jiwer':version('jiwer'), 'synthesis_contract_sha256':sha(directory/'contract.json'),
                    'scoring':'fr-runtime-2 casefold; Unicode word tokens; apostrophes/hyphens split; accents retained'}
        if (directory/'scoring_contract.json').exists():
            if json.loads((directory/'scoring_contract.json').read_text()) != contract:
                raise ValueError('Scoring contract changed')
        else:
            write(directory/'scoring_contract.json', contract)
        log_path = directory/'scores.jsonl'
        previous = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()] if log_path.exists() else []
        done = {row['id'] for row in previous}
        with log_path.open('a', encoding='utf-8') as log:
            for index,row in enumerate(rows):
                if row['id'] in done:
                    continue
                if args.follow:
                    waited = time.monotonic()
                    while True:
                        ready = set()
                        for line in (directory/'synthesis.jsonl').read_text().splitlines():
                            try:
                                ready.add(json.loads(line)['id'])
                            except json.JSONDecodeError:
                                pass  # Producer may be writing the final log line.
                        if row['id'] in ready:
                            break
                        if time.monotonic()-waited > 600:
                            raise TimeoutError('No new synthesized sentence for ten minutes; resume scoring after synthesis')
                        time.sleep(1)
                audio, rate = sf.read(directory/(row['id']+'.wav'), dtype='float32')
                common = math.gcd(rate,16000)
                audio = resample_poly(audio,16000//common,rate//common).astype(np.float32)
                segments, _ = asr.transcribe(audio, language='fr', beam_size=c['beam_size'],
                    temperature=0, condition_on_previous_text=False, vad_filter=False)
                hypothesis = ' '.join(segment.text.strip() for segment in segments)
                reference = word_text(row['normalized_text'])
                normalized_hypothesis = word_text(hypothesis)
                measures = jiwer.process_words(reference, normalized_hypothesis)
                result = {'id':row['id'], 'reference':reference, 'transcript':hypothesis,
                    'normalized_transcript':normalized_hypothesis, 'wer':measures.wer,
                    'substitutions':measures.substitutions, 'deletions':measures.deletions,
                    'insertions':measures.insertions, 'reference_words':len(reference.split()),
                    'dnsmos_p835':dns_score(audio)}
                log.write(json.dumps(result,ensure_ascii=False)+'\n')
                log.flush()
                if (index+1)%10 == 0:
                    print(f'SCORED {variant} {index+1}/{len(rows)} WER={result["wer"]:.3f}', flush=True)
        write(directory/'scoring_complete.json', {'status':'COMPLETE', 'count':len(rows)})


def aggregate(args, config, rows):
    report = {'status':'COMPLETE', 'count':len(rows), 'corpus_sha256':sha(args.corpus),
        'config':config, 'quality_proxy_limit':config['dnsmos']['interpretation'], 'variants':{},
        'timing_contention':'Multiple synthesis/ASR CPU tasks and independent GPU training shared this laptop; not an isolated speed ranking.'}
    lengths = np.array([len(word_text(row['normalized_text']).split()) for row in rows])
    report['corpus_statistics'] = {'reference_word_count_min':int(lengths.min()),
        'reference_word_count_max':int(lengths.max()),'reference_word_count_mean':float(lengths.mean()),
        'normalized_characters_min':min(len(row['normalized_text']) for row in rows),
        'normalized_characters_max':max(len(row['normalized_text']) for row in rows),
        'length_bins':json.loads(args.corpus.with_suffix('.audit.json').read_text())['length_bins']}
    report['model_inventory'] = json.loads((ROOT/'evaluation/model_inventory.json').read_text())
    report['thread_configuration'] = 'ORT intra_op=2, inter_op=1; Torch intra_op=2. Auxiliary numerical libraries use their defaults.'
    for variant in args.variants:
        directory = args.output/variant
        synth = json.loads((directory/'synthesis_report.json').read_text(encoding='utf-8'))
        timings = [json.loads(line) for line in (directory/'synthesis.jsonl').read_text().splitlines()]
        scores = [json.loads(line) for line in (directory/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        scoring = json.loads((directory/'scoring_contract.json').read_text(encoding='utf-8'))
        expected = {row['id'] for row in rows}
        if len(timings)!=len(rows) or len(scores)!=len(rows) or {s['id'] for s in scores}!=expected or {s['id'] for s in timings}!=expected:
            raise ValueError('Evaluation coverage incomplete or duplicated')
        sums = {key:sum(row[key] for row in scores) for key in ('substitutions','deletions','insertions','reference_words')}
        rng = np.random.default_rng(config['seed'])
        errors = np.array([row['substitutions']+row['deletions']+row['insertions'] for row in scores])
        counts = np.array([row['reference_words'] for row in scores])
        draws = rng.integers(0,len(scores),(2000,len(scores)))
        interval = np.percentile(errors[draws].sum(1)/counts[draws].sum(1),[2.5,97.5]).tolist()
        report['variants'][variant] = {'count':len(rows), 'wer':sum(sums[k] for k in ('substitutions','deletions','insertions'))/sums['reference_words'],
            'wer_bootstrap_95_percent_interval':interval,
            **sums, 'dnsmos_p835':{key:float(np.mean([row['dnsmos_p835'][key] for row in scores])) for key in ('sig','bak','ovrl')},
            'rtf_median':float(np.median([row['rtf'] for row in timings])),
            'rtf_p95':float(np.percentile([row['rtf'] for row in timings],95)),
            'rtf_global':sum(row['elapsed_seconds'] for row in timings)/sum(row['seconds'] for row in timings),
            'generated_seconds':sum(row['seconds'] for row in timings), 'synthesis_seconds':sum(row['elapsed_seconds'] for row in timings),
            'duration_min_seconds':min(row['seconds'] for row in timings),
            'duration_max_seconds':max(row['seconds'] for row in timings),
            'mean_generation_seconds':float(np.mean([row['elapsed_seconds'] for row in timings])),
            'max_clipped_fraction':max(row['clipped_fraction'] for row in timings),
            'scoring_contract':scoring, **synth}
    write(args.output/'report.json', report)
    print(json.dumps({'status':report['status'],'count':report['count'],
        'report':str(args.output/'report.json'),'variants':{key:{name:row[name] for name in
            ('wer','dnsmos_p835','rtf_median','peak_process_rss_bytes')} for key,row in report['variants'].items()}},
        ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('synthesize','score','aggregate'))
    parser.add_argument('--config', type=Path, default=ROOT/'evaluation/config_v0.json')
    parser.add_argument('--corpus', type=Path, default=ROOT/'evaluation/heldout_v0.jsonl')
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/private/release_v0/exports')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/private/release_v0/evaluation')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--variant', choices=('fp32','int8','kokoro','piper'), default='fp32')
    parser.add_argument('--variants', nargs='+', default=['fp32','int8','kokoro','piper'])
    parser.add_argument('--follow', action='store_true', help='Score committed WAVs while synthesis is still running')
    parser.add_argument('--asr-device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--asr-compute', choices=('int8','float16','int8_float16'), default='int8')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 300:
        raise ValueError('Use all 300 frozen held-out sentences')
    {'synthesize':synthesize, 'score':score, 'aggregate':aggregate}[args.stage](args,config,rows)


if __name__ == '__main__':
    main()
