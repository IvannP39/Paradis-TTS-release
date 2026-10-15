"""French student TTS with an offline phase 7 reference pipeline."""

__all__ = ['ParadisTTS', 'OnnxTTS', 'PostProcessing', 'soften_audio', 'save_wav', 'postprocess_wav',
           'download_model', '__version__']


def __getattr__(name):
    if name in ('ParadisTTS', '__version__', 'download_model'):
        from paradis_tts import api
        return getattr(api,name)
    if name=='OnnxTTS':
        from paradis_tts.onnx_runtime import OnnxTTS
        return OnnxTTS
    if name in ('PostProcessing','soften_audio','save_wav','postprocess_wav'):
        from paradis_tts import audio
        return getattr(audio,name)
    if name=='SynthesisResult':
        from paradis_tts.inference import ParadisTTS, SynthesisResult
        return {'ParadisTTS': ParadisTTS, 'SynthesisResult': SynthesisResult}[name]
    raise AttributeError(name)
