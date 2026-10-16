"""Offline ONNX speech/WAV CLI and standalone gentle postprocessing."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from paradis_tts.audio import PostProcessing,read_wav,save_wav,soften_audio


def main():
    parser=argparse.ArgumentParser(prog='python -m paradis_tts')
    commands=parser.add_subparsers(dest='command',required=True)
    synth=commands.add_parser('synthesize',help='Generate an actual 24 kHz WAV using an ONNX bundle')
    synth.add_argument('--model-folder','--bundle',dest='model_folder',type=Path)
    synth.add_argument('--variant',choices=('fp32','int8'),default='fp32')
    synth.add_argument('--weight-kind',choices=('ema','raw'),help='Default: the release-selected weights (EMA in 0.1.2)')
    synth.add_argument('--no-post-traitement',action='store_true')
    synth.add_argument('--quiet',action='store_true',help='Disable initialization messages and download progress')
    text=synth.add_mutually_exclusive_group(required=True)
    text.add_argument('--text');text.add_argument('--text-file',type=Path)
    synth.add_argument('--speed',type=float,default=1.)
    synth.add_argument('--seed',type=int,default=1729)
    synth.add_argument('--threads',type=int,default=2)
    post=commands.add_parser('postprocess',help='Soften an existing mono 24 kHz PCM16 WAV')
    post.add_argument('input',type=Path)
    for command in (synth,post):
        command.add_argument('--output',type=Path,required=True)
        command.add_argument('--settings',type=Path,help='Standard settings or browser slider JSON')
        command.add_argument('--overwrite',action='store_true')
        for option in ('dynamic-db','upper-db','attack-db','syllable-db','smoothing'):
            command.add_argument('--'+option,type=float)
    args=parser.parse_args()
    settings=PostProcessing.from_file(args.settings) if args.settings else None
    aliases={'dynamic_db':'dynamic_soften_db','upper_db':'upper_soften_db',
             'attack_db':'attack_soften_db','syllable_db':'syllable_soften_db','smoothing':'dynamics_smoothing'}
    overrides={target:getattr(args,source) for source,target in aliases.items() if getattr(args,source) is not None}
    if args.command=='synthesize' and args.no_post_traitement and (settings is not None or overrides):
        parser.error('--no-post-traitement cannot be combined with processing settings')
    if args.command=='synthesize':
        from paradis_tts.api import ParadisTTS
        model=ParadisTTS(model_folder=args.model_folder,threads=args.threads,postprocessing=settings,
                         variant=args.variant,weight_kind=args.weight_kind,post_traitement=not args.no_post_traitement,
                         verbose=not args.quiet)
        model.postprocessing=replace(model.postprocessing,**overrides)
        text=args.text if args.text is not None else args.text_file.read_text(encoding='utf-8-sig')
        result=model.to_file(text,args.output,seed=args.seed,speed=args.speed,overwrite=args.overwrite)
        print(json.dumps({'output':str(args.output),'normalized_text':result.normalized_text,**result.audit()},ensure_ascii=False))
    else:
        settings=replace(settings or PostProcessing(),**overrides)
        signal=read_wav(args.input)
        audio=soften_audio(signal,dynamic_db=settings.dynamic_soften_db,upper_db=settings.upper_soften_db,
            attack_db=settings.attack_soften_db,syllable_db=settings.syllable_soften_db,
            smoothing=settings.dynamics_smoothing,match_level=settings.softening_match_level)
        save_wav(audio,args.output,overwrite=args.overwrite)
        metadata={'input':str(args.input),'stage':'softening_only','phase_filter_applied':False,
            'settings':replace(settings,phase_lock_frames=0,presence_restore=False).to_dict(),
            'samples':len(audio),'sample_rate':24000}
        args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'output':str(args.output),'sample_rate':24000,'samples':len(audio)}))


if __name__=='__main__': main()
