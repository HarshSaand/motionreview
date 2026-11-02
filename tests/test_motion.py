import csv,io,json,unittest
from pathlib import Path
import numpy as np
from motionreview.pose import NAMES,PARENTS,parse_bvh,read_pose,normalize,resample
from motionreview.inference import analyse,csv_export
ROOT=Path(__file__).resolve().parents[1]
def fixture():return json.loads((ROOT/'artifacts/example_pose.json').read_text())
class PoseTests(unittest.TestCase):
 def test_example_has_real_source(self):
  d=fixture();self.assertIn('una-dinosauria',d['source']);self.assertEqual(len(d['source_sha256']),64);self.assertEqual(read_pose(json.dumps(d),'json').shape[1:],(17,3))
 def test_joint_schema_rejected(self):
  d=fixture();d['joint_names'][0]='unknown'
  with self.assertRaises(ValueError):read_pose(json.dumps(d),'json')
 def test_nonfinite_rejected(self):
  d=fixture();d['positions'][0][0][0]=float('nan')
  with self.assertRaises(ValueError):read_pose(json.dumps(d),'json')
 def test_invalid_fps_and_empty(self):
  for fps in [0,True,float('inf'),10000]:
   with self.assertRaises(ValueError):resample(np.ones((20,17,3)),fps)
  with self.assertRaises(ValueError):read_pose('{}','json')
 def test_normalization_translation_invariant(self):
  x=np.array(fixture()['positions'][:15]);a,s=normalize(x);b,t=normalize(x+123);np.testing.assert_allclose(a,b,atol=1e-6);self.assertAlmostEqual(s,t)
 def test_zero_bones_rejected(self):
  with self.assertRaises(ValueError):normalize(np.zeros((15,17,3)))
 def test_bvh_forward_kinematics(self):
  def node(i):
   rows=[('ROOT ' if i==0 else 'JOINT ')+NAMES[i],'{','OFFSET 0 '+('0' if i==0 else '1')+' 0','CHANNELS '+('6 Xposition Yposition Zposition Zrotation Yrotation Xrotation' if i==0 else '3 Zrotation Yrotation Xrotation')]
   for j,p in enumerate(PARENTS):
    if p==i:rows+=node(j)
   return rows+['}']
  vals=[1,0,0,90,0,0]+[0]*48;text='HIERARCHY\n'+'\n'.join(node(0))+'\nMOTION\nFrames: 15\nFrame Time: 0.0333333333333333\n'+'\n'.join(' '.join(map(str,vals)) for _ in range(15))
  p=parse_bvh(text);np.testing.assert_allclose(p[0,0],[1,0,0],atol=1e-5);np.testing.assert_allclose(p[0,1],[0,0,0],atol=1e-5)
 def test_session_splits_disjoint(self):
  m=json.loads((ROOT/'artifacts/data_manifest.json').read_text());sets={s:{r['session'] for r in m if r['split']==s} for s in ['train','validation','test']};self.assertFalse(sets['train']&sets['test']);self.assertFalse(sets['train']&sets['validation']);self.assertFalse(sets['validation']&sets['test'])
class InferenceTests(unittest.TestCase):
 def test_future_cannot_change_prediction(self):
  d=fixture();a=analyse(json.dumps(d),'json',30);arr=np.array(d['positions']);arr[30:,1:]*=2;d['positions']=arr.tolist();b=analyse(json.dumps(d),'json',30);np.testing.assert_array_equal(a['forecast']['mlp'],b['forecast']['mlp']);self.assertNotEqual(a['observed_error_pct_body_scale']['mlp'],b['observed_error_pct_body_scale']['mlp'])
 def test_unseen_end_has_no_invented_score(self):
  r=analyse(json.dumps(fixture()),'json');self.assertEqual(r['observed_future_frames'],0);self.assertEqual(r['observed_error_pct_body_scale'],{});self.assertEqual(np.array(r['forecast']['mlp']).shape,(15,17,3))
 def test_export_rows(self):
  r=analyse(json.dumps(fixture()),'json',30);rows=list(csv.DictReader(io.StringIO(csv_export(r))));self.assertEqual(len(rows),765);self.assertEqual(set(x['method'] for x in rows),{'mlp','persistence','constant_velocity'})
 def test_invalid_cutoff(self):
  for c in [1,True,30.5,999999]:
   with self.assertRaises(ValueError):analyse(json.dumps(fixture()),'json',c)

class ProvenanceTests(unittest.TestCase):
 def test_modified_download_rejected(self):
  import importlib.util,tempfile,hashlib
  spec=importlib.util.spec_from_file_location('download',ROOT/'scripts/download_data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'data').mkdir();(root/'artifacts').mkdir();(root/'data/02_01.bvh').write_bytes(b'modified')
   (root/'artifacts/data_manifest.json').write_text(json.dumps([{'file':'02_01.bvh','sha256':hashlib.sha256(b'original').hexdigest()}]));module.ROOT=root
   with self.assertRaisesRegex(ValueError,'Checksum mismatch'):module.main()
