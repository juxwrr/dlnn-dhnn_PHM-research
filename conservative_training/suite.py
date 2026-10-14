"""Durable, sequential execution; full physical training remains audit-gated."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def now():return datetime.now(ZoneInfo('America/Phoenix')).isoformat()


def execute(root):
    root=Path(root).resolve();jobs=root/'jobs';jobs.mkdir(exist_ok=True)
    code_root=Path(__file__).resolve().parent.parent
    status_path=jobs/'status.json'
    state={'started':now(),'pid':os.getpid(),'status':'running','completed_tasks':[],
           'physical_full_training':'gated: unresolved proper-time/field action and cable material bounds',
           'final_test_used_for_checkpoint_selection':False,
           'scientific_results_complete':False}
    def status(**changes):
        state.update(changes);state['updated']=now()
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status_path)
        report=['# Conservative LNN execution status','',f'Updated: {state["updated"]}',
                '',f'Status: **{state["status"]}**',f'Current task: {state.get("current_task","none")}',
                '',f'Physical full model: {state["physical_full_training"]}',
                '', 'No complete two-model scientific result is claimed. Existing results and synced references are unchanged.',
                '', 'Completed tasks:']+[f'- {x}' for x in state['completed_tasks']]
        if state.get('reason'):report+=['',state['reason']]
        (root/'LIVE_STATUS.md').write_text('\n'.join(report)+'\n')
    def wait_for(run):
        status(current_task='waiting for smoke test',current_run=str(run))
        while not (run/'final.pkl').exists():
            if (run/'failure.json').exists():raise RuntimeError(f'Smoke test failed: {run}')
            time.sleep(10)
    def run_task(name,module,args):
        command=[sys.executable,'-m',module]+[str(x) for x in args]
        log=jobs/(name+'.log')
        status(current_task=name,log=str(log),command=command)
        with log.open('x') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,cwd=code_root)
            status(child_pid=child.pid)
            result=child.wait()
        if result:raise RuntimeError(f'{name} failed with exit {result}; see {log}')
        state['completed_tasks'].append(name);status(child_pid=None)
    def training(name,model,seed,steps=300000,resume=None,baseline=False,low=False):
        args=['--model',model,'--seed',seed,'--steps',steps,'--out',root/'runs'/name]
        if model=='source':args+=['--optimizer','faithful']
        if model=='full_effective':args+=['--microbatch',16]
        if resume:args+=['--resume',resume]
        if baseline:args+=['--baseline']
        if low:args+=['--low-speed']
        run_task(name,'conservative_training.train',args)
    try:
        source=root/'runs/source_float32_smoke_seed0';incomplete=root/'runs/incomplete_smoke_seed0'
        wait_for(source);wait_for(incomplete)
        audit=json.loads((root/'audit_complete/physics_audit.json').read_text(encoding='utf-8'))
        if not audit['incomplete_reference_gate_passed']:raise RuntimeError('Incomplete reference gate failed')
        if not audit['full_training_gates']['default_pinv_retains_nine_modes']:raise RuntimeError('Joint numerical gate failed')
        # Establish full-budget source behavior before substituting PHM dynamics.
        training('source_full_seed0','source',0,resume=source/'final.pkl')
        run_task('source_full_evaluation','conservative_training.evaluate',[root/'runs/source_full_seed0/best.pkl'])
        control=json.loads((root/'runs/source_full_seed0/final_evaluation/metrics.json').read_text(encoding='utf-8'))
        original=next(x for x in control['rollouts'] if x['name']=='notebook_seed4')
        score=json.loads((root/'runs/source_full_seed0/history.jsonl').read_text(encoding='utf-8').splitlines()[-1])['validation_score']
        healthy=score<.5 and not original['failed'] and original['beta_MAE']<.05
        status(source_control_diagnostic_gate={'passed':healthy,'validation_score':score,
                    'notebook_rollout':original,'criteria':'fixed validation weighted MAE < 0.5 and notebook-seed4 beta MAE < .05c; diagnostic project gates, not paper tolerances'})
        if not healthy:
            status(status='blocked_source_control',reason='Full-budget source control did not recover the predeclared diagnostic behavior; PHM long runs not started')
            return
        # Joint smoke is a benchmark, never a high-speed physical spacecraft result.
        training('full_balanced_smoke_seed0','full_effective',0,steps=1000)
        for seed in range(5):
            name=f'incomplete_full_seed{seed}'
            try:
                training(name,'incomplete',seed,resume=incomplete/'final.pkl' if seed==0 else None)
                run_task(name+'_evaluation','conservative_training.evaluate',[root/'runs'/name/'best.pkl'])
            except RuntimeError as error:
                state.setdefault('seed_failures',[]).append({'experiment':name,'seed':seed,'reason':str(error)})
                status()
        # Matched direct acceleration baseline, same streamed data and update budget.
        for seed in range(5):
            name=f'incomplete_baseline_full_seed{seed}'
            try:
                training(name,'incomplete',seed,baseline=True)
                run_task(name+'_evaluation','conservative_training.evaluate',[root/'runs'/name/'best.pkl'])
            except RuntimeError as error:
                state.setdefault('seed_failures',[]).append({'experiment':name,'seed':seed,'reason':str(error)})
                status()
        # Independent low-speed-only models; final evaluation remains high-speed stratified.
        for seed in range(5):
            name=f'incomplete_low_speed_seed{seed}'
            try:
                training(name,'incomplete',seed,low=True)
                run_task(name+'_evaluation','conservative_training.evaluate',[root/'runs'/name/'best.pkl'])
            except RuntimeError as error:
                state.setdefault('seed_failures',[]).append({'experiment':name,'seed':seed,'reason':str(error)})
                status()
        status(status='incomplete_study_finished_full_model_gated',current_task=None,
               reason='Full physical action remains unresolved; full-model final runs, full baseline, coupled rollouts and interaction ladder remain outstanding')
    except BaseException as error:
        status(status='failed',reason=f'{type(error).__name__}: {error}',child_pid=state.get('child_pid'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    execute(parser.parse_args().root)
