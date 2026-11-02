import json,threading,unittest,urllib.request,urllib.error
from motionreview.app import server
from test_motion import fixture
class AppTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=server(0);cls.t=threading.Thread(target=cls.s.serve_forever,daemon=True);cls.t.start();cls.url=f'http://127.0.0.1:{cls.s.server_port}'
 @classmethod
 def tearDownClass(cls):cls.s.shutdown();cls.s.server_close();cls.t.join()
 def post(self,data,headers=None):
  req=urllib.request.Request(self.url+'/api/analyse',json.dumps(data).encode(),headers={'Content-Type':'application/json',**(headers or {})});return json.load(urllib.request.urlopen(req))
 def test_custom_upload_and_export(self):
  d=fixture();d={k:d[k] for k in ['joint_names','fps','positions']}
  for frame in d['positions']:
   for point in frame:
    for j in range(3):point[j]=point[j]*.1+4
  r=self.post({'text':json.dumps(d),'kind':'json','cutoff':30});self.assertEqual(len(r['forecast']['mlp']),15);self.assertIn('x_body_scale',r['csv'])
 def test_bad_input_returns_400(self):
  with self.assertRaises(urllib.error.HTTPError) as e:self.post({'text':'{}','kind':'json'})
  self.assertEqual(e.exception.code,400)
 def test_cross_origin_rejected(self):
  with self.assertRaises(urllib.error.HTTPError) as e:self.post({'text':'{}','kind':'json'},{'Origin':'https://example.org'})
  self.assertEqual(e.exception.code,400)
 def test_example_available(self):
  d=json.load(urllib.request.urlopen(self.url+'/api/example'));self.assertIn('positions',d)

 def test_oversized_body_rejected_before_read(self):
  import http.client
  client=http.client.HTTPConnection('127.0.0.1',self.s.server_port,timeout=3)
  try:
   client.request('POST','/api/analyse',body=b'{}',headers={'Content-Type':'application/json','Content-Length':str(22*1024*1024)})
   response=client.getresponse();self.assertEqual(response.status,400);self.assertIn(b'exceeds',response.read())
  finally:client.close()
