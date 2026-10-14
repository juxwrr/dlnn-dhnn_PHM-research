"""One study postprocessing job; creates original-derivation plots after primary runs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def execute(root):
    root=Path(root).resolve();jobs=root/'jobs';code=Path(__file__).resolve().parent.parent
    state=dict(pid=os.getpid(),status='waiting_for_primary_full_checkpoints',completed_seeds=[],failed_seeds=[],
               physical_full_claim=False,current_seed=None,current_child=None,updated=None,
               reference='PHM_Derivations.pdf plus archived original parameter implementation')
    path=jobs/'original_comparison_status.json'
    def status():
        state['updated']=datetime.now(ZoneInfo('America/Phoenix')).isoformat()
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    status()
    try:
        while len(state['completed_seeds'])+len(state['failed_seeds'])<5:
            for seed in range(5):
                if seed in state['completed_seeds'] or seed in [entry['seed'] for entry in state['failed_seeds']]:continue
                one=root/f'runs/incomplete_full_seed{seed}';joint=root/f'runs/coupled_math_main_seed{seed}'
                if any((run/'failure.json').exists() for run in [one,joint]):
                    state['failed_seeds'].append(dict(seed=seed,reason='Primary training failed; no complete checkpoint pair is available'));status();continue
                if not all((run/'runtime.json').exists() and (run/'best.pkl').exists() for run in [one,joint]):continue
                out=root/f'comparisons/original_derivation_full_seed{seed}'
                if out.exists():raise FileExistsError(f'Preserve existing result: {out}')
                command=[sys.executable,'-m','conservative_training.original_dissipative_comparison',
                         '--incomplete',str(one/'best.pkl'),'--coupled',str(joint/'best.pkl'),'--out',str(out)]
                state.update(status='evaluating',current_seed=seed);status()
                with (jobs/f'original_derivation_full_seed{seed}.log').open('x') as log:
                    child=subprocess.Popen(command,cwd=code,stdout=log,stderr=subprocess.STDOUT)
                    state['current_child']=child.pid;status();result=child.wait()
                state['current_child']=None
                if result:state['failed_seeds'].append(dict(seed=seed,reason=f'Comparison failed with exit {result}; inspect its log'))
                else:state['completed_seeds'].append(seed)
                state.update(status='waiting_for_primary_full_checkpoints',current_seed=None);status()
            for name in ['status.json','mathematical_benchmark_status.json']:
                task=json.loads((jobs/name).read_text(encoding='utf-8'))
                if task['status'] in ['failed','blocked_source_control']:
                    state.update(status='blocked_by_training_suite',reason=f'{name}: {task.get("reason",task["status"])}');status();return
            time.sleep(60);status()
        state.update(status='completed',current_seed=None);status()
    except BaseException as error:
        state.update(status='failed',reason=f'{type(error).__name__}: {error}');status();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    execute(parser.parse_args().root)
