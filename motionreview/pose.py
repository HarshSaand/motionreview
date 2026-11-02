from __future__ import annotations
import json,re
import numpy as np
NAMES=['Hips','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot','Spine','Spine1','Neck','Head','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand']
PARENTS=[-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]
MAX_FRAMES=20000

def rotation(axis,angle):
    c=np.cos(angle);s=np.sin(angle);out=np.zeros((len(angle),3,3));out[:,range(3),range(3)]=1
    i,j={'X':(1,2),'Y':(2,0),'Z':(0,1)}[axis]
    out[:,i,i]=c;out[:,j,j]=c;out[:,i,j]=-s;out[:,j,i]=s
    return out

def parse_bvh(text,drop_reference=False):
    if len(text.encode())>20*1024*1024:raise ValueError('Maximum BVH size is 20 MiB')
    if 'MOTION' not in text:raise ValueError('BVH needs HIERARCHY and MOTION')
    hierarchy,motion=text.split('MOTION',1);tokens=re.findall(r'[^\s{}]+|[{}]',hierarchy);pos=0;nodes=[];channels=[]
    def take():
        nonlocal pos
        if pos>=len(tokens):raise ValueError('Incomplete hierarchy')
        x=tokens[pos];pos+=1;return x
    if take()!='HIERARCHY' or take()!='ROOT':raise ValueError('Invalid BVH header')
    def node(name,parent,depth=0):
        if depth>64 or len(nodes)>100:raise ValueError('Hierarchy too large')
        idx=len(nodes);nodes.append(dict(name=name,parent=parent,offset=None,channels=[]))
        if take()!='{':raise ValueError('Expected hierarchy brace')
        while True:
            token=take()
            if token=='}':break
            if token=='OFFSET':nodes[idx]['offset']=np.array([float(take()) for _ in range(3)])
            elif token=='CHANNELS':
                count=int(take())
                if not 0<=count<=6:raise ValueError('Unsupported channel count')
                for _ in range(count):
                    ch=take()
                    if ch not in ['Xposition','Yposition','Zposition','Xrotation','Yrotation','Zrotation']:raise ValueError('Unsupported channel')
                    nodes[idx]['channels'].append((ch,len(channels)));channels.append(ch)
            elif token=='JOINT':node(take(),idx,depth+1)
            elif token=='End':
                if take()!='Site':raise ValueError('Invalid End Site')
                node(f'_end_{idx}',idx,depth+1)
            else:raise ValueError('Unsupported hierarchy field: '+token)
        if nodes[idx]['offset'] is None or not np.isfinite(nodes[idx]['offset']).all():raise ValueError('Missing/nonfinite offset')
    try:node(take(),-1)
    except (IndexError,OverflowError,RecursionError) as e:raise ValueError('Invalid hierarchy') from e
    lines=motion.strip().splitlines()
    if len(lines)<3:raise ValueError('Motion is empty')
    try:count=int(lines[0].split(':')[1]);dt=float(lines[1].split(':')[1])
    except (ValueError,IndexError):raise ValueError('Invalid motion header') from None
    if not 15<=count<=MAX_FRAMES or not np.isfinite(dt) or not 1/240<=dt<=1/15:raise ValueError('Require 15–20,000 frames at 15–240 Hz')
    try:values=np.loadtxt(lines[2:],dtype=np.float64,ndmin=2)
    except ValueError:raise ValueError('Invalid motion values') from None
    if values.shape!=(count,len(channels)) or not np.isfinite(values).all() or np.abs(values).max()>1e6:raise ValueError('Inconsistent or unbounded motion values')
    rotations=[];positions=[]
    for n in nodes:
        local=np.broadcast_to(np.eye(3),(count,3,3)).copy();offset=np.broadcast_to(n['offset'],(count,3)).copy()
        for ch,index in n['channels']:
            if ch.endswith('position'):offset[:,'XYZ'.index(ch[0])]+=values[:,index]
            else:local=local@rotation(ch[0],np.deg2rad(values[:,index]))
        p=n['parent']
        positions.append(offset if p<0 else positions[p]+np.einsum('nij,nj->ni',rotations[p],offset))
        rotations.append(local if p<0 else rotations[p]@local)
    lookup={n['name']:i for i,n in enumerate(nodes)}
    missing=set(NAMES)-lookup.keys()
    if missing:raise ValueError('Missing supported joints: '+', '.join(sorted(missing)))
    xyz=np.stack([positions[lookup[name]] for name in NAMES],axis=1)
    if drop_reference:xyz=xyz[1:]
    return resample(xyz,1/dt)

def resample(xyz,fps):
    if not isinstance(fps,(float,int)) or isinstance(fps,bool) or not np.isfinite(fps) or not 15<=fps<=240:raise ValueError('fps must be between 15 and 240')
    xyz=np.asarray(xyz,dtype=np.float32)
    if xyz.ndim!=3 or xyz.shape[1:]!=(17,3) or not 15<=len(xyz)<=MAX_FRAMES or not np.isfinite(xyz).all() or np.abs(xyz).max()>1e6:raise ValueError('Need 15–20,000 finite frames shaped [frames,17,3]')
    times=np.arange(len(xyz))/fps;target=np.arange(0,times[-1]+1e-8,1/30)
    out=np.stack([np.interp(target,times,x) for x in xyz.reshape(len(xyz),-1).T],axis=1).reshape(-1,17,3)
    if len(out)<15:raise ValueError('At least 0.5 seconds of observed motion is required')
    return out.astype(np.float32)

def read_pose(text,kind):
    if len(text.encode())>20*1024*1024:raise ValueError('File exceeds 20 MiB')
    if kind=='bvh':return parse_bvh(text)
    if kind!='json':raise ValueError('Upload .bvh or .json')
    try:d=json.loads(text)
    except (ValueError,RecursionError):raise ValueError('Invalid pose JSON') from None
    if not isinstance(d,dict) or d.get('joint_names')!=NAMES:raise ValueError('JSON joint_names must match the downloadable template')
    try:return resample(d['positions'],d['fps'])
    except (KeyError,TypeError,OverflowError):raise ValueError('JSON requires finite positions and fps') from None

def normalize(history):
    # Body scale uses observed bone lengths only, never future frames.
    pairs=[(0,1),(1,2),(2,3),(0,7),(7,8),(8,9),(9,10)]
    scale=float(np.median(sum(np.linalg.norm(history[:,a]-history[:,b],axis=1) for a,b in pairs)))
    if not np.isfinite(scale) or scale<1e-6:raise ValueError('Skeleton bone scale is zero or invalid')
    return (history-history[:,:1])/scale,scale
