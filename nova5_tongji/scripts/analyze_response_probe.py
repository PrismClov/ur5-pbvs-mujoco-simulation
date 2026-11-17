import json,pathlib,sys,numpy as np
def rotvec(A):
 A=np.asarray(A);v=.5*np.stack([A[...,2,1]-A[...,1,2],A[...,0,2]-A[...,2,0],A[...,1,0]-A[...,0,1]],axis=-1)
 sn=np.linalg.norm(v,axis=-1);cs=np.clip((np.trace(A,axis1=-2,axis2=-1)-1)/2,-1,1)
 angle=np.arctan2(sn,cs);return v*(angle/np.maximum(sn,1e-12))[...,None]
root=pathlib.Path(sys.argv[1]);rows=[json.loads(x) for x in (root/'response_probe.jsonl').read_text().splitlines()]
t=np.array([r['elapsed'] for r in rows]);u=np.array([r['reference_pitch_rad'] for r in rows]);R=np.array([r['current_pose'] for r in rows])[:,:3,:3];D=np.array([r['desired_pose'] for r in rows])[:,:3,:3]
def meanrot(A):
 U,_,V=np.linalg.svd(np.mean(A,axis=0));return U@np.diag([1,1,np.linalg.det(U@V)])@V
metrics=[]
for edge in np.flatnonzero(np.diff(u)!=0)+1:
 te=t[edge];end=min(te+7.8,t[-1]);mask=(t>=te)&(t<=end);before=(t>=te-.5)&(t<te)
 if end-te<7 or np.sum(before)<5:continue
 r0=meanrot(R[before]);d0=meanrot(D[before]);d1=meanrot(D[(t>=te+.1)&(t<te+.5)])
 delta=rotvec(d1@d0.T);amplitude=np.linalg.norm(delta);axis=delta/amplitude
 y=rotvec(R[mask]@r0.T)@axis*180/np.pi;tt=t[mask]-te
 # Relative-to-achieved rise time is distinct from achieving the one-degree request.
 final=float(np.median(y[tt>6.5]));target=float(amplitude*180/np.pi)
 def crossing(level):
  for i in range(len(y)-2):
   if np.all(y[i:i+3]>=level):return float(tt[i])
  return None
 remain=np.array([r['rotation_error_deg'] for r in rows])[mask]
 metrics.append(dict(edge_s=float(te),command_pitch_deg=float(u[edge]*180/np.pi),
  desired_delta_deg=target,achieved_delta_deg=final,first_0_1deg_s=crossing(.1),
  reach_90pct_command_s=crossing(.9*target),reach_90pct_achieved_s=crossing(.9*final),
  response_at_200ms_deg=float(np.interp(.2,tt,y)),response_at_300ms_deg=float(np.interp(.3,tt,y)),
  maximum_delta_deg=float(max(y)),final_error_deg=float(np.median(remain[tt>6.5]))))
result=dict(run=root.name,gain=rows[0]['gain'],duration_s=float(t[-1]),steps=metrics)
(root/'response_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
