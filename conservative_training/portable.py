"""Portable fresh training of both models, using the unchanged study trainer."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parent.parent
MODELS={'one-term':'incomplete','coupled':'full_effective'}


def configure(device):
    if device=='cpu':os.environ['JAX_PLATFORMS']='cpu'
    elif device=='gpu':os.environ['JAX_PLATFORMS']='rocm'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    devices=jax.devices()
    if device=='gpu' and not any(d.platform!='cpu' for d in devices):
        raise RuntimeError('AMD GPU requested but JAX has no ROCm GPU device. CPU fallback is disabled.')
    return {'jax_version':jax.__version__,'devices':[str(d) for d in devices],'backend':jax.default_backend()}


def verify_transfer():
    folder=REPO/'transfer'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name,item in manifest['runs'].items():
        for file,key in [('resume.pkl','resume_sha256'),('best.pkl','best_sha256'),('validation.npz','validation_sha256')]:
            if hashlib.sha256((folder/name/file).read_bytes()).hexdigest()!=item[key]:
                raise ValueError(f'Transfer checksum mismatch: {name}/{file}')
    gate=json.loads((folder/'source_gate.json').read_text(encoding='utf-8'))
    if not gate['passed']:raise RuntimeError('Transferred source-control gate did not pass')
    return manifest


def checkpoint_for(run, model, seed, experiment, use_transfer):
    paths=list(run.glob('checkpoint_*.pkl'))
    paths.extend(run/name for name in ['interrupted.pkl','best.pkl'] if (run/name).exists())
    if paths:
        # The interrupted state can be newer than the last periodic checkpoint.
        return max(paths,key=lambda file:pickle.loads(file.read_bytes())['step'])
    if use_transfer and seed==0 and experiment=='main':
        name='incomplete_full_seed0' if model=='one-term' else 'coupled_math_main_seed0'
        return REPO/'transfer'/name/'resume.pkl'
    return None


def train_arguments(model,seed,out,resume=None,experiment='main',steps=300000,profile=False):
    arguments=['--model',MODELS[model],'--seed',str(seed),'--steps',str(steps),'--out',str(out),
               '--microbatch','16' if model=='coupled' else '32']
    if model=='coupled':arguments+=['--mathematical-benchmark','--physics-audit',str(REPO/'conservative_training/audit_complete/physics_audit.json')]
    if resume:arguments+=['--resume',str(resume)]
    if experiment=='baseline':arguments+=['--baseline']
    if experiment=='low_speed':arguments+=['--low-speed']
    if profile:arguments+=['--profile']
    return arguments


def run_study(args):
    devices=configure(args.device)
    if not args.fresh:verify_transfer()
    else:
        gate=json.loads((REPO/'evidence/source_control/source_gate.json').read_text(encoding='utf-8'))
        if not gate['passed']:raise RuntimeError('Source gate did not pass')
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    status_path=root/'study_status.json'
    active_child=[None]
    state={'started':datetime.now(timezone.utc).isoformat(),'devices':devices,'status':'running',
           'primary_seeds':args.seeds,'completed':[],'failures':[],'physical_full_claim':False,
           'budget':300000,'initialization':'fresh weights for new runs' if args.fresh else 'optional transferred seed-zero checkpoints',
           'backend_change_note':'Local interrupted runs retain optimizer/RNG state; different hardware may introduce floating-point differences'}
    def save():
        state['updated']=datetime.now(timezone.utc).isoformat()
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(status_path)
    def task(name,module,arguments):
        log=root/(name+'.log');state.update(current_task=name,log=str(log));save()
        command=[sys.executable,'-m',module]+list(map(str,arguments))
        print(f'Starting {name}. Detailed log: {log}',flush=True)
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
            active_child[0]=child
            state['child_pid']=child.pid;save()
            previous=None
            while child.poll() is None:
                progress=next((Path(str(value))/'progress.json' for value in arguments if Path(str(value)).is_dir() and (Path(str(value))/'progress.json').exists()),None)
                if progress:
                    try:
                        update=json.loads(progress.read_text(encoding='utf-8'))
                        if update['step']!=previous:
                            print(f'{name}: {update["step"]:,} / {update["total"]:,} updates',flush=True);previous=update['step']
                    except (ValueError,OSError):pass
                time.sleep(5)
            if child.returncode:raise RuntimeError(f'{name} failed; see {log}')
        active_child[0]=None
        state['child_pid']=None;state['completed'].append(name);save()
    try:
        model_names=list(MODELS) if args.model=='both' else [args.model]
        experiments=['main','baseline','low_speed'] if args.controls else ['main']
        for experiment in experiments:
            for seed in args.seeds:
                for model in model_names:
                    name=f'{model}_{experiment}_seed{seed}';run=root/name
                    try:
                        completed=(run/'runtime.json').exists() and json.loads((run/'runtime.json').read_text(encoding='utf-8')).get('scientific_result',False)
                        if not completed:
                            resume=checkpoint_for(run,model,seed,experiment,not args.fresh)
                            task(name,'conservative_training.train',train_arguments(model,seed,run,resume,experiment))
                        evaluation=run/'final_evaluation'
                        if not evaluation.exists():
                            task(name+'_evaluation','conservative_training.evaluate_coupled' if model=='coupled' else 'conservative_training.evaluate',[run/'best.pkl'])
                        elif not (evaluation/'metrics.json').exists():
                            raise RuntimeError(f'Partial evaluation exists at {evaluation}; preserve it and rerun with a new --label')
                    except (RuntimeError,ValueError) as error:
                        state['failures'].append({'task':name,'reason':str(error)});save()
                if experiment=='main' and args.model=='both':
                    one=root/f'one-term_main_seed{seed}';joint=root/f'coupled_main_seed{seed}';out=root/f'original_comparison_seed{seed}'
                    if all((r/'runtime.json').exists() and json.loads((r/'runtime.json').read_text(encoding='utf-8')).get('scientific_result',False) for r in [one,joint]) and not out.exists():
                        try:task(f'original_comparison_seed{seed}','conservative_training.original_dissipative_comparison',['--incomplete',one/'best.pkl','--coupled',joint/'best.pkl','--out',out])
                        except RuntimeError as error:state['failures'].append({'task':'original_comparison','seed':seed,'reason':str(error)});save()
        state.update(status='completed_with_failures' if state['failures'] else 'completed',current_task=None);save()
    except KeyboardInterrupt:
        # Both processes receive Ctrl+C on Windows/Linux. Allow the trainer to save its committed state.
        if active_child[0] is not None:
            print('Waiting for the trainer to save its interrupted checkpoint...',flush=True)
            active_child[0].wait()
        state.update(status='interrupted');save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('check');check.add_argument('--device',choices=['cpu','gpu'],default='cpu');check.add_argument('--transfer',action='store_true')
    study=sub.add_parser('study');study.add_argument('--model',choices=['one-term','coupled','both'],default='both');study.add_argument('--device',choices=['cpu','gpu'],default='cpu');study.add_argument('--out',default='training_results');study.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));study.add_argument('--controls',action='store_true');study.add_argument('--fresh',action='store_true',default=True,help='Start new runs from freshly initialized weights (default); interrupted local runs can resume');study.add_argument('--resume-transfer',dest='fresh',action='store_false',help='Optionally continue archived seed-zero snapshots instead')
    profile=sub.add_parser('profile');profile.add_argument('--model',choices=list(MODELS),required=True);profile.add_argument('--device',choices=['cpu','gpu'],default='cpu');profile.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='check':
        print(json.dumps(configure(args.device),indent=2))
        if args.transfer:print(json.dumps(verify_transfer()['runs'],indent=2))
    elif args.command=='profile':
        configure(args.device)
        subprocess.run([sys.executable,'-m','conservative_training.train']+train_arguments(args.model,0,Path(args.out),steps=5,profile=True),cwd=REPO,check=True)
    else:
        if len(set(args.seeds))!=len(args.seeds) or any(s not in range(5) for s in args.seeds):parser.error('Choose distinct primary seeds from 0 through 4')
        run_study(args)


if __name__=='__main__':main()
