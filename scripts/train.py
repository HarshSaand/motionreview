import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from motionreview.pose import parse_bvh,normalize,NAMES,PARENTS
from motionreview.model import ForecastNet
ROOT=Path(__file__).resolve().parents[1]
def windows(pose,stride):
    xs=[];ys=[]
    for start in range(0,len(pose)-29,stride):
        hist=pose[start:start+15];x,scale=normalize(hist);future=pose[start+15:start+30]
        y=(future-future[:,:1])/scale
        xs.append(x.reshape(15,51));ys.append(y.reshape(15,51))
    return np.array(xs,dtype='float32'),np.array(ys,dtype='float32')
def error(a,b):return float(np.linalg.norm((a-b).reshape(-1,15,17,3)[:,:,1:],axis=-1).mean()*100)
def main():
    torch.set_num_threads(4);torch.manual_seed(41);np.random.seed(41)
    manifest=json.loads((ROOT/'artifacts/data_manifest.json').read_text());sets={k:[] for k in ['train','validation','test']};meta=[]
    for item in manifest:
        p=ROOT/'data'/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        pose=parse_bvh(p.read_text(),drop_reference=True);x,y=windows(pose,3 if item['split']=='train' else 15)
        if not len(x):raise ValueError('Capture too short '+item['file'])
        sets[item['split']].append((x,y,item['file']));meta.append(dict(file=item['file'],split=item['split'],frames_30hz=len(pose),windows=len(x)))
    tx=torch.tensor(np.concatenate([x for x,y,n in sets['train']]));ty=torch.tensor(np.concatenate([y for x,y,n in sets['train']]))
    vx=torch.tensor(np.concatenate([x for x,y,n in sets['validation']]));vy=np.concatenate([y for x,y,n in sets['validation']])
    model=ForecastNet();optimizer=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');trace=[]
    for epoch in range(35):
        model.train();indices=torch.randperm(len(tx));losses=[]
        for ids in indices.split(128):
            optimizer.zero_grad();prediction=model(tx[ids]);loss=((prediction-ty[ids])**2).mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():ve=error(model(vx).numpy(),vy)
        trace.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),validation_error_pct_body_scale=ve))
        if ve<best:best=ve;torch.save(model.state_dict(),ROOT/'artifacts/forecast.pt');best_epoch=epoch+1
    model.load_state_dict(torch.load(ROOT/'artifacts/forecast.pt',weights_only=True));model.eval()
    metrics={};predictions={}
    for split in ['validation','test']:
        rows=[]
        for x,y,name in sets[split]:
            with torch.no_grad():learned=model(torch.tensor(x)).numpy()
            hold=np.repeat(x[:,-1:],15,axis=1);v=(x[:,-1]-x[:,-5])/4;linear=x[:,-1:]+np.arange(1,16)[None,:,None]*v[:,None]
            rows.append(dict(file=name,windows=len(x),mlp=error(learned,y),persistence=error(hold,y),constant_velocity=error(linear,y)))
        metrics[split]=dict(sequence_mean={k:float(np.mean([r[k] for r in rows])) for k in ['mlp','persistence','constant_velocity']},sequences=rows,windows=sum(r['windows'] for r in rows))
    selected=min(metrics['validation']['sequence_mean'],key=metrics['validation']['sequence_mean'].get)
    report=dict(dataset='CMU optical motion capture, Hahne BVH conversion',captures=len(manifest),seed=41,epochs=35,best_epoch=best_epoch,input_frames=15,horizon_frames=15,fps=30,unit='mean non-root joint position error as percent of observed body scale',splits=metrics,train_windows=len(tx),selected_method=selected,selection='lowest validation sequence-mean error; no test selection',limitations=['32-capture locomotion pilot','session IDs disjoint; identity separation not guaranteed by CMU','root-relative pose only, not global translation, force or collision prediction','no physical robot or clinical validation'])
    (ROOT/'artifacts/metrics.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'artifacts/training_trace.json').write_text(json.dumps(trace,indent=2)+'\n');(ROOT/'artifacts/window_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    # Derived real held-out example, retaining provenance, no artificial T pose.
    item=next(i for i in manifest if i['split']=='test');pose=parse_bvh((ROOT/'data'/item['file']).read_text(),drop_reference=True)
    example=dict(joint_names=NAMES,fps=30,positions=pose[:120].tolist(),source=item['url'],source_sha256=item['sha256'],note='Derived FK joint positions; artificial first reference frame removed.')
    (ROOT/'artifacts/example_pose.json').write_text(json.dumps(example)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
