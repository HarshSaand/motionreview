import csv,io,json
from pathlib import Path
import numpy as np
import torch
from .pose import NAMES,PARENTS,normalize,read_pose
from .model import ForecastNet
ROOT=Path(__file__).resolve().parents[1]

def analyse(text,kind,cutoff=None):
    xyz=read_pose(text,kind)
    if cutoff is None:cutoff=len(xyz)
    if isinstance(cutoff,bool) or not isinstance(cutoff,int) or not 15<=cutoff<=len(xyz):raise ValueError('Forecast frame must be from 15 to the end of the clip')
    history=xyz[cutoff-15:cutoff];norm,scale=normalize(history);x=norm.reshape(1,15,51)
    model=ForecastNet();model.load_state_dict(torch.load(ROOT/'artifacts/forecast.pt',map_location='cpu',weights_only=True));model.eval()
    with torch.no_grad():pred=model(torch.tensor(x)).numpy().reshape(15,17,3)
    hold=np.repeat(norm[-1:],15,axis=0);velocity=(norm[-1]-norm[-5])/4;linear=norm[-1]+np.arange(1,16)[:,None,None]*velocity
    pred[:,0]=0  # Root-relative output has an exact zero root; benchmark excludes root.
    methods=dict(mlp=pred,persistence=hold,constant_velocity=linear)
    ground=xyz[cutoff:cutoff+15];truth=(ground-ground[:,:1])/scale if len(ground) else np.empty((0,17,3))
    scores={k:float(np.linalg.norm(v[:len(truth),1:]-truth[:,1:],axis=-1).mean()*100) for k,v in methods.items()} if len(truth) else {}
    metrics=json.loads((ROOT/'artifacts/metrics.json').read_text())
    result=dict(joint_names=NAMES,parents=PARENTS,fps=30,cutoff_frame=cutoff,total_frames=len(xyz),history=norm.tolist(),forecast={k:v.tolist() for k,v in methods.items()},observed_future=truth.tolist(),observed_future_frames=len(truth),observed_error_pct_body_scale=scores,selected_method=metrics['selected_method'],input_body_scale=scale,coordinate_system='root-relative; units are observed body scale',note='A 0.5-second pose forecast, not motion capture from video or a physical motion controller.')
    # Alert if learned forecast changes bone lengths appreciably vs last observed pose.
    last=np.array([np.linalg.norm(norm[-1,j]-norm[-1,p]) for j,p in enumerate(PARENTS) if p>=0]);future=np.stack([np.linalg.norm(pred[:,j]-pred[:,p],axis=1) for j,p in enumerate(PARENTS) if p>=0],axis=1)
    valid=last>.005  # Ignore collapsed connector joints, which have no stable relative length.
    result['bone_length_edges_evaluated']=int(valid.sum())
    result['max_mlp_bone_length_change_pct']=float(np.max(np.abs(future[:,valid]-last[valid])/last[valid])*100) if valid.any() else None
    return result

def csv_export(result):
    out=io.StringIO();w=csv.writer(out);w.writerow(['method','future_frame','seconds','joint','x_body_scale','y_body_scale','z_body_scale'])
    for method,frames in result['forecast'].items():
        for frame,pose in enumerate(frames,1):
            for name,point in zip(NAMES,pose):w.writerow([method,frame,frame/30,name,*point])
    return out.getvalue()
