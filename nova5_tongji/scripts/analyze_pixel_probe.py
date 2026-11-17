"""Report actual image-error recovery; timestamps remain host arrival proxies."""
import json
from pathlib import Path
import sys
import numpy as np

def analyze(path):
    rows=[json.loads(s) for s in (path/'pixel_probe.jsonl').read_text().splitlines()]
    recovery=[r for r in rows if r['phase'] in ('recover','hold')]
    log=(path/'runtime.log').read_text(encoding='utf-8',errors='replace') if (path/'runtime.log').exists() else ''
    faults=[s for s in log.splitlines() if s.startswith('RuntimeError:')]
    if not recovery:return dict(run=path.name,result='no_recovery_test',fault=faults[-1] if faults else None)
    t=np.array([r['t']-r['return_time'] for r in recovery])
    e=np.array([r['center_error_px'] for r in recovery]);norm=np.linalg.norm(e,axis=1)
    def sustained(band,duration=.10):
        for i in range(len(t)):
            end=np.searchsorted(t,t[i]+duration)
            if end<len(t) and np.all(norm[i:end+1]<=band):return float(t[i])
        return None
    return dict(run=path.name,gain=recovery[0]['gain'],
        initial_error_px=float(norm[0]),
        error_at_200ms_px=float(np.interp(.2,t,norm)) if t[-1]>=.2 else None,
        first_10pct_reduction_s=sustained(.9*norm[0]),
        enter_5px_for_300ms_s=sustained(5,.3),
        enter_10pct_for_300ms_s=sustained(.1*norm[0],.3),
        opposite_side_overshoot_px=float(max(0,-e[:,0].min())),
        final_error_px=float(norm[-1]),
        max_reference_joint_speed_rad_s=max(float(np.linalg.norm(r.get('qdot_raw',[0]))) for r in recovery),
        result='aborted' if faults else recovery[-1]['result'],fault=faults[-1] if faults else None,
        timing='Control host time paired with image-arrival proxy, not calibrated capture time',
        tolerance='Within 5 px continuously for 300 ms; reported time is band entry')

if __name__=='__main__':
    root=Path(sys.argv[1]);result=analyze(root)
    (root/'pixel_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
