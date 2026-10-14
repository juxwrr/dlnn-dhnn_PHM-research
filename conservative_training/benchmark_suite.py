"""Durable extension of the existing suite; no running source snapshot is edited."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from .train import record_json


def execute(root):
    root=Path(root).resolve(); jobs=root/'jobs'; code=Path(__file__).resolve().parent.parent
    status_path=jobs/'mathematical_benchmark_status.json'
    state=dict(pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),status='waiting_for_source_control',
               physical_full_claim=False,physical_full_training='gated',completed_tasks=[],seed_failures=[],
               training_budget_per_run=300000,primary_coupled_seeds=list(range(5)),scientific_results_complete=False,
               one_term_suite_status=str(jobs/'status.json'),code_snapshot=str(code),
               user_authorization='Train the mathematical benchmark while keeping physical claims gated',
               current_task=None,child_pid=None)
    def status(**updates):
        state.update(updates);state['updated']=datetime.now(timezone.utc).isoformat()
        temp=status_path.with_suffix('.tmp');record_json(temp,state);temp.replace(status_path)
    def task(name,module,args):
        command=[sys.executable,'-m',module]+list(map(str,args));log=jobs/(name+'.log')
        status(current_task=name,command=command,log=str(log),status='running')
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=code,stdout=stream,stderr=subprocess.STDOUT)
            status(child_pid=child.pid)
            result=child.wait()
        status(child_pid=None)
        if result:raise RuntimeError(f'Exit {result}; see {log}')
        state['completed_tasks'].append(name);status()
    def update_one_term_charts():
        # This watcher only reads completed full-budget runs. It never uses their tests to select checkpoints.
        while True:
            for seed in range(5):
                run=root/f'runs/incomplete_full_seed{seed}';out=root/f'comparisons/one_term_full_seed{seed}'
                if (run/'runtime.json').exists() and (run/'best.pkl').exists() and not out.exists():
                    log=jobs/f'one_term_full_seed{seed}_dissipative_chart.log'
                    with log.open('a') as stream:
                        subprocess.run([sys.executable,'-m','conservative_training.dissipative_comparison',str(run/'best.pkl'),'--out',str(out)],cwd=code,stdout=stream,stderr=subprocess.STDOUT,check=False)
            time.sleep(30)
    threading.Thread(target=update_one_term_charts,daemon=True).start()
    status()
    try:
        while True:
            original=json.loads((jobs/'status.json').read_text(encoding='utf-8'))
            gate=original.get('source_control_diagnostic_gate')
            if gate:
                if not gate['passed']:
                    status(status='blocked_source_control',reason='The predeclared source-control gate failed. Coupled long runs have not started.');return
                break
            if original['status'] in ['failed','blocked_source_control']:
                raise RuntimeError('Existing suite stopped before a healthy source control; inspect its status and logs')
            time.sleep(30)
        audit=root/'audit_complete/physics_audit.json'
        checked=json.loads(audit.read_text(encoding='utf-8'))
        if not checked['full_training_gates']['default_pinv_retains_nine_modes']:
            raise RuntimeError('Numerical nine-mode audit failed')
        smoke=root/'runs/full_balanced_smoke_seed0'
        while not (smoke/'runtime.json').exists():
            if (smoke/'failure.json').exists():
                raise RuntimeError('Joint smoke failed; long coupled training requires diagnosis first')
            status(status='waiting_for_joint_smoke',current_task='full_balanced_smoke_seed0')
            time.sleep(30)
        # The original coordinator continues the five-seed one-term study and controls.
        # This extension runs the coupled study in parallel; total wall time reflects CPU sharing.
        for experiment in ['main','baseline','low_speed']:
            for seed in range(5):
                name=f'coupled_math_{experiment}_seed{seed}';run=root/'runs'/name
                args=['--model','full_effective','--seed',seed,'--steps',300000,'--microbatch',16,
                      '--mathematical-benchmark','--physics-audit',audit,'--out',run]
                if experiment=='main' and seed==0:args+=['--resume',smoke/'final.pkl']
                if experiment=='baseline':args+=['--baseline']
                if experiment=='low_speed':args+=['--low-speed']
                try:
                    task(name,'conservative_training.train',args)
                    task(name+'_evaluation','conservative_training.evaluate_coupled',[run/'best.pkl'])
                except RuntimeError as error:
                    state['seed_failures'].append(dict(experiment=experiment,seed=seed,reason=str(error)));status()
        status(status='training_and_bounded_evaluation_finished',current_task=None,
               reason='No complete physical success claim: physical audit, five-seed report, interaction ladder and verified mission multirate resolution remain outstanding')
    except BaseException as error:
        status(status='failed',reason=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    execute(parser.parse_args().root)
