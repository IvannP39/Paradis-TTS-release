"""Offline French ONNX inference: three neural graphs, NumPy source and postprocessing."""
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from paradis_tts.audio import PostProcessing,process_audio,soften_audio,save_wav
from paradis_tts.frontend import FrenchFrontend
from paradis_tts.normalization_v2 import normalize_sentence,VERSION


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InferenceFrontend(FrenchFrontend):
    def prepare(self,text):
        normalized,rules=normalize_sentence(text)
        normalized,extra,chunks=super().prepare(normalized)
        return normalized,list(dict.fromkeys(rules+extra)),chunks


def harmonic_source(f0,parameters,seed):
    """Kokoro-style harmonic excitation; explicit PCG64 noise for repeatable ORT A/B.

    Runtime noise is deterministic for a seed, but is not Torch's RNG sequence.
    Interpolation follows align_corners=False, and source mixing uses saved weights.
    """
    f0=np.asarray(f0,dtype=np.float32).reshape(-1)
    count=len(f0);scale=int(parameters['upsample_scale']);n=count*scale
    harmonics=np.arange(1,int(parameters['dim'])+1,dtype=np.float32)
    increments=np.remainder(f0[:,None]*harmonics[None]/np.float32(24000),1)
    phase=np.cumsum(increments,axis=0,dtype=np.float64).astype(np.float32)
    phase=phase*np.float32(2*np.pi)*np.float32(scale)
    positions=(np.arange(n,dtype=np.float64)+.5)/scale-.5
    left=np.floor(positions).astype(np.int64);mix=(positions-left).astype(np.float32)[:,None]
    low=phase[np.clip(left,0,count-1)];high=phase[np.clip(left+1,0,count-1)]
    phase=low*(1-mix)+high*mix
    voiced=(np.repeat(f0,scale)>parameters['voiced_threshold']).astype(np.float32)[:,None]
    generator=np.random.default_rng(int(seed))
    noise=generator.standard_normal((n,len(harmonics)),dtype=np.float32)
    noise*=voiced*parameters['noise_std']+(1-voiced)*parameters['sine_amp']/3
    sine=np.sin(phase)*parameters['sine_amp']*voiced+noise
    weight=np.asarray(parameters['weight'],dtype=np.float32).reshape(-1,1)
    bias=np.asarray(parameters['bias'],dtype=np.float32)
    return np.tanh(sine@weight+bias).astype(np.float32)[None]


@dataclass
class Speech:
    audio:np.ndarray
    normalized_text:str
    chunks:list
    elapsed_seconds:float
    seed:int
    postprocessing:dict
    sample_rate:int=24000

    @property
    def seconds(self): return len(self.audio)/self.sample_rate

    def audit(self):
        return {'seconds':self.seconds,'elapsed_seconds':self.elapsed_seconds,
            'rtf':self.elapsed_seconds/self.seconds,'peak':float(np.max(np.abs(self.audio))),
            'rms':float(np.sqrt(np.mean(self.audio.astype(np.float64)**2))),
            'clipped_fraction':float(np.mean(np.abs(self.audio)>1))}

    def save(self,path,*,overwrite=False):
        path=save_wav(self.audio,path,overwrite=overwrite)
        metadata={'normalized_text':self.normalized_text,'seed':self.seed,'sample_rate':self.sample_rate,
            'chunks':self.chunks,'postprocessing':self.postprocessing,**self.audit()}
        path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return path


class OnnxTTS:
    def __init__(self,bundle,*,threads=2,postprocessing=None):
        import onnxruntime as ort
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/'manifest.json').read_text(encoding='utf-8'))
        if (self.manifest.get('schema_version')!=1 or self.manifest.get('normalizer_version')!=VERSION
            or self.manifest.get('sample_rate')!=24000):
            raise ValueError('Incompatible Paradis ONNX bundle')
        if not isinstance(threads,int) or threads<1: raise ValueError('threads must be a positive integer')
        for name,digest in self.manifest['files'].items():
            path=(self.bundle/name).resolve()
            if not path.is_relative_to(self.bundle) or sha256(path)!=digest: raise ValueError(f'Checksum mismatch: {name}')
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.sessions={name:ort.InferenceSession(str(self.bundle/f'{name}.onnx'),options,
            providers=['CPUExecutionProvider']) for name in ('text','prosody','decoder')}
        self.frontend=InferenceFrontend(json.loads((self.bundle/'vocab.json').read_text(encoding='utf-8')))
        actual=list(self.frontend.g2p.backend.version())
        if actual!=self.manifest['espeak_version']: raise ValueError('Use the eSpeak version recorded in the export')
        settings_path=self.bundle/'postprocessing.json'
        self.postprocessing=postprocessing or (PostProcessing.from_file(settings_path) if settings_path.exists()
                                               else PostProcessing(**self.manifest['inference_defaults']))
        if not isinstance(self.postprocessing,PostProcessing): raise ValueError('Expected PostProcessing settings')

    def synthesize(self,text,*,seed=1729,speed=1.):
        if not np.isfinite(speed) or not .5<=speed<=2: raise ValueError('speed must be between 0.5 and 2')
        if not isinstance(seed,int) or seed<0: raise ValueError('seed must be a nonnegative integer')
        started=time.perf_counter();normalized,rules,chunks=self.frontend.prepare(text)
        signals=[];metadata=[]
        phase_settings=replace(self.postprocessing,dynamic_soften_db=0,upper_soften_db=0,
            attack_soften_db=0,syllable_soften_db=0,dynamics_smoothing=0)
        for index,chunk in enumerate(chunks):
            ids=np.array([chunk.ids],dtype=np.int64)
            durations,prosody,features=self.sessions['text'].run(None,{'input_ids':ids})
            if not all(np.isfinite(value).all() for value in (durations,prosody,features)):
                raise FloatingPointError('Invalid text predictions')
            counts=np.maximum(1,np.rint(durations[0]/speed)).astype(np.int64)
            frames=int(counts.sum())
            if frames>4096: raise ValueError('Predicted duration exceeds 4096-frame segment budget')
            token_index=np.repeat(np.arange(len(counts)),counts)
            expanded=np.ascontiguousarray(prosody[:,:,token_index],dtype=np.float32)
            asr=np.ascontiguousarray(features[:,:,token_index],dtype=np.float32)
            f0,noise=self.sessions['prosody'].run(None,{'expanded':expanded})
            if not np.isfinite(f0).all() or not np.isfinite(noise).all(): raise FloatingPointError('Invalid acoustic predictions')
            source=harmonic_source(f0,self.manifest['source'],seed+index)
            audio=self.sessions['decoder'].run(None,{'features':asr,'f0':f0,'noise':noise,'source':source})[0][0]
            if len(audio)!=frames*600: raise ValueError('Decoder output length differs')
            audio=process_audio(audio,source=source.reshape(-1),f0=f0.reshape(-1),settings=phase_settings)
            if index: signals.append(np.zeros(3600,dtype=np.float32))
            signals.append(audio)
            metadata.append({'text':chunk.text,'phonemes':chunk.phonemes,'frames':frames,'seed':seed+index})
        settings=self.postprocessing
        audio=soften_audio(np.concatenate(signals),dynamic_db=settings.dynamic_soften_db,
            upper_db=settings.upper_soften_db,attack_db=settings.attack_soften_db,
            syllable_db=settings.syllable_soften_db,smoothing=settings.dynamics_smoothing,
            match_level=settings.softening_match_level)
        if float(np.sqrt(np.mean(audio.astype(np.float64)**2)))<1e-7:
            raise FloatingPointError('Silent student waveform')
        return Speech(audio,normalized,metadata,time.perf_counter()-started,seed,settings.to_dict())

    def __call__(self,text,**kwargs): return self.synthesize(text,**kwargs).audio

    def save(self,audio,path,**kwargs): return save_wav(audio,path,**kwargs)

    def to_file(self,text,path,*,seed=1729,speed=1.,overwrite=False):
        result=self.synthesize(text,seed=seed,speed=speed)
        result.save(path,overwrite=overwrite)
        return result
