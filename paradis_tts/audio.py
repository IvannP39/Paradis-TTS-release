"""Portable 24 kHz postprocessing and PCM WAV output. NumPy only; no Torch."""
from dataclasses import dataclass,asdict
import math
from pathlib import Path
import json
import wave
import numpy as np

RATE=24000


def mono(audio):
    values=np.asarray(audio,dtype=np.float32)
    if values.ndim!=1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite mono audio')
    return values


def stft(audio,n=1024,hop=256):
    audio=mono(audio)
    if len(audio)<=n//2: raise ValueError('Audio too short for this STFT')
    padded=np.pad(audio,(n//2,n//2),mode='reflect')
    frames=np.lib.stride_tricks.sliding_window_view(padded,n)[::hop]
    window=(.5-.5*np.cos(2*np.pi*np.arange(n)/n)).astype(np.float32)
    return np.fft.rfft(frames*window,axis=-1).T.astype(np.complex64)


def istft(spectrum,length,n=1024,hop=256):
    window=(.5-.5*np.cos(2*np.pi*np.arange(n)/n)).astype(np.float32)
    frames=np.fft.irfft(spectrum.T,n=n,axis=-1)*window
    size=(frames.shape[0]-1)*hop+n
    output=np.zeros(size,dtype=np.float64);norm=np.zeros(size,dtype=np.float64)
    for index,frame in enumerate(frames):
        start=index*hop
        output[start:start+n]+=frame
        norm[start:start+n]+=window.astype(np.float64)**2
    return (output[n//2:n//2+length]/np.maximum(norm[n//2:n//2+length],1e-12)).astype(np.float32)


def phase_lock(audio,source,f0,length=97):
    if length not in (9,17,33,49,65,97): raise ValueError('Unsupported phase filter length')
    audio=mono(audio);source=mono(source);f0=mono(f0)
    if source.shape!=audio.shape: raise ValueError('Excitation/audio lengths differ')
    spectrum=stft(audio);excitation=stft(source)
    frequencies=np.arange(513)*RATE/1024
    band=((frequencies>=2000)&(frequencies<8000))[:,None]
    voiced=(f0[np.minimum(np.arange(spectrum.shape[1])*256//300,len(f0)-1)]>60)[None]
    relative=spectrum*excitation.conj()/np.maximum(np.abs(excitation),1e-6)
    kernel=(.5-.5*np.cos(2*np.pi*np.arange(length+2)/(length+2)))[1:-1].astype(np.float32)
    kernel/=kernel.sum()
    padded=np.pad(relative,((0,0),(length//2,length//2)))
    smoothed=np.stack([np.convolve(row,kernel[::-1],mode='valid') for row in padded])
    angles=np.where(band&voiced,np.angle(excitation)+np.angle(smoothed),np.angle(spectrum))
    return istft(np.abs(spectrum)*np.exp(1j*angles),len(audio))


def rise(frequencies,lo,hi):
    return .5-.5*np.cos(np.pi*np.clip((frequencies-lo)/(hi-lo),0,1))


def restore_presence(raw,filtered,max_gain_db=2.5):
    raw=mono(raw);filtered=mono(filtered)
    frequencies=np.arange(513)*RATE/1024
    before=np.mean(np.abs(stft(raw))**2,axis=-1)
    after=np.mean(np.abs(stft(filtered))**2,axis=-1)
    gains=[]
    for lo,hi in ((2000,4000),(4000,8000)):
        band=(frequencies>=lo)&(frequencies<hi)
        gain=10*np.log10(max(float(before[band].sum()),1e-12)/max(float(after[band].sum()),1e-12))
        gains.append(np.clip(gain,0,max_gain_db))
    n=1<<(2*len(raw)-1).bit_length();f=np.fft.rfftfreq(n,1/RATE)
    weight=rise(f,1500,2000)*(1-rise(f,3500,4500))*gains[0]
    weight+=rise(f,3500,4500)*(1-rise(f,7500,8500))*gains[1]
    return np.fft.irfft(np.fft.rfft(filtered,n)*10**(weight/20),n)[:len(raw)].astype(np.float32)


def dynamics(audio,attack_db,syllable_db,smoothing):
    if not (attack_db or syllable_db) or not np.any(audio): return audio.copy()
    values=audio.astype(np.float64);prefix=np.r_[0.,np.cumsum(values**2)]
    centers=np.arange(0,len(audio),24);left=np.maximum(0,centers-60);right=np.minimum(len(audio),centers+60)
    rms=np.sqrt(np.maximum(0,(prefix[right]-prefix[left])/(right-left)))
    floor=max(float(rms.max())*10**(-35/20),1e-12)
    active=np.sort(rms[rms>=floor]);threshold=(active[int(.8*(len(active)-1))] if len(active) else floor)*10**(3/20)
    slow=max(float(rms[0]),floor);alpha=math.exp(-.001/.08)
    attacks=[];syllables=[]
    for value in rms:
        excess=20*math.log10(max(float(value),floor)/max(slow,floor))
        attacks.append(min(attack_db,max(0.,(excess-3)*.5)))
        slow=alpha*slow+(1-alpha)*value
        over=20*math.log10(max(float(value),1e-12)/max(float(threshold),1e-12))
        cut=0. if over<=-3 else ((over+3)**2/36 if over<3 else over/3)
        syllables.append(min(syllable_db,cut) if value>=floor else 0.)
    def smooth(desired,attack,release):
        previous=0.;output=[]
        for index in range(len(desired)):
            value=desired[min(index+4,len(desired)-1)]
            coefficient=math.exp(-.001/(attack if value>previous else release))
            previous=coefficient*previous+(1-coefficient)*value;output.append(previous)
        output=np.array(output)
        radius=math.ceil(24*smoothing)
        if radius:
            weights=np.hanning(2*radius+3)[1:-1];weights/=weights.sum()
            output=np.convolve(np.pad(output,radius,mode='edge'),weights,mode='valid')
        return output
    cut=smooth(attacks,.002,.035)+smooth(syllables,.008,.1)
    envelope=np.interp(np.arange(len(values)),centers,cut)
    return (values*10**(-envelope/20)).astype(np.float32)


@dataclass(frozen=True)
class PostProcessing:
    phase_lock_frames:int=97
    presence_restore:bool=True
    presence_max_gain_db:float=2.5
    dynamic_soften_db:float=1.
    upper_soften_db:float=1.5
    attack_soften_db:float=0.
    syllable_soften_db:float=0.
    dynamics_smoothing:float=0.
    softening_match_level:bool=True

    def __post_init__(self):
        for name,maximum in (('presence_max_gain_db',3),('dynamic_soften_db',2),('upper_soften_db',2),
                             ('attack_soften_db',3),('syllable_soften_db',3),('dynamics_smoothing',1)):
            value=getattr(self,name)
            if not math.isfinite(value) or not 0<=value<=maximum: raise ValueError(f'Invalid {name}')
        if self.phase_lock_frames not in (0,9,17,33,49,65,97): raise ValueError('Invalid phase filter length')
        if type(self.presence_restore) is not bool or type(self.softening_match_level) is not bool:
            raise ValueError('Expected boolean processing switches')
        if self.presence_restore and not self.phase_lock_frames: raise ValueError('Presence requires phase correction')

    def to_dict(self): return asdict(self)

    @classmethod
    def from_file(cls,path):
        value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        value=value.get('postprocessing',value)
        aliases={'dynamic_db':'dynamic_soften_db','upper_db':'upper_soften_db',
            'attack_db':'attack_soften_db','syllable_db':'syllable_soften_db',
            'match_level':'softening_match_level'}
        metadata={'reference','order','level_matching'}
        settings={}
        for name,item in value.items():
            if name in metadata: continue
            name=aliases.get(name,name)
            if name not in cls.__dataclass_fields__: raise ValueError(f'Unknown processing setting: {name}')
            if name in settings: raise ValueError(f'Duplicate processing setting: {name}')
            settings[name]=item
        return cls(**settings)

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2)+'\n',encoding='utf-8')


def read_wav(path):
    with wave.open(str(path),'rb') as reader:
        if (reader.getnchannels(),reader.getframerate(),reader.getsampwidth())!=(1,RATE,2):
            raise ValueError('Expected a mono PCM16 WAV at 24000 Hz')
        return np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768


def soften_audio(audio,*,dynamic_db=1.,upper_db=1.5,attack_db=0.,syllable_db=0.,smoothing=0.,match_level=True):
    """Apply the user-selected softening to an existing mono 24 kHz waveform."""
    PostProcessing(dynamic_soften_db=dynamic_db,upper_soften_db=upper_db,
        attack_soften_db=attack_db,syllable_soften_db=syllable_db,dynamics_smoothing=smoothing,
        softening_match_level=match_level)
    reference=mono(audio)
    if not any((dynamic_db,upper_db,attack_db,syllable_db)) or not np.any(reference): return reference.copy()
    output=reference.copy()
    if dynamic_db:
        spectrum=stft(output,hop=128);f=np.arange(513)*RATE/1024
        band=rise(f,3000,4000)*(1-rise(f,8000,10000))
        power=np.abs(spectrum)**2
        total=np.maximum(power[(f>=300)&(f<10000)].sum(0),1e-12)
        desired=np.clip(((power*band[:,None]).sum(0)/total-.25)/.35,0,1)*dynamic_db
        desired*=total>=total.max()*1e-4
        previous=0.;envelope=[]
        for value in desired:
            alpha=math.exp(-(128/RATE)/(.005 if value>previous else .04))
            previous=alpha*previous+(1-alpha)*value;envelope.append(previous)
        output=istft(spectrum*10**(-band[:,None]*np.array(envelope)[None]/20),len(output),hop=128)
    if upper_db:
        n=1<<(2*len(output)-1).bit_length();f=np.fft.rfftfreq(n,1/RATE)
        gain=10**(-rise(f,8000,10000)*upper_db/20)
        output=np.fft.irfft(np.fft.rfft(output,n)*gain,n)[:len(output)].astype(np.float32)
    output=dynamics(output,attack_db,syllable_db,smoothing)
    before=np.sqrt(np.mean(reference.astype(np.float64)**2));after=np.sqrt(np.mean(output.astype(np.float64)**2))
    if match_level and min(before,after)>1e-8:
        gain=float(np.clip(20*np.log10(before/after),-.3,.3))
        peak=float(np.max(np.abs(output)))
        if peak: gain=min(gain,20*np.log10(.999/peak))
        output=(output*10**(gain/20)).astype(np.float32)
    return mono(output)


def process_audio(audio,*,source=None,f0=None,settings=None):
    """Full student postprocessing; phase correction requires the matching excitation."""
    settings=settings or PostProcessing()
    raw=mono(audio);output=raw
    if settings.phase_lock_frames:
        if source is None or f0 is None: raise ValueError('Matching source and F0 are required for phase correction')
        output=phase_lock(raw,source,f0,settings.phase_lock_frames)
        if settings.presence_restore: output=restore_presence(raw,output,settings.presence_max_gain_db)
    return soften_audio(output,dynamic_db=settings.dynamic_soften_db,upper_db=settings.upper_soften_db,
        attack_db=settings.attack_soften_db,syllable_db=settings.syllable_soften_db,
        smoothing=settings.dynamics_smoothing,match_level=settings.softening_match_level)


def save_wav(audio,path,*,sample_rate=RATE,overwrite=False):
    """Write an actual mono PCM16 WAV, refusing nonfinite values or clipping."""
    audio=mono(audio);path=Path(path)
    if path.exists() and not overwrite: raise FileExistsError(path)
    if float(np.max(np.abs(audio)))>1: raise ValueError('Audio exceeds PCM range; apply explicit peak normalization')
    if sample_rate!=RATE: raise ValueError('Paradis audio must be 24000 Hz')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with wave.open(str(temporary),'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(sample_rate)
        writer.writeframes(np.rint(audio*32767).astype('<i2').tobytes())
    temporary.replace(path)
    return path


def postprocess_wav(source,destination,*,overwrite=False,**settings):
    """One-call softening for an existing mono PCM16 24 kHz WAV."""
    return save_wav(soften_audio(read_wav(source),**settings),destination,overwrite=overwrite)
