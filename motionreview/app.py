"""Local pose upload and forecast review application."""
import argparse,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .inference import analyse,csv_export
ROOT=Path(__file__).resolve().parents[1]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def local(self):
        valid={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in valid:raise ValueError('Use the local app address')
        if self.headers.get('Origin') and self.headers['Origin'] not in {'http://'+h for h in valid}:raise ValueError('Cross-origin request refused')
    def send(self,data,status=200,kind='application/json'):
        data=json.dumps(data,allow_nan=False).encode() if kind=='application/json' else data.encode()
        self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        try:
            self.local();p=urlsplit(self.path).path
            if p=='/':self.send((ROOT/'motionreview/app.html').read_text(),kind='text/html')
            elif p=='/api/example':self.send(json.loads((ROOT/'artifacts/example_pose.json').read_text()))
            elif p=='/api/metrics':self.send(json.loads((ROOT/'artifacts/metrics.json').read_text()))
            else:self.send({'error':'Not found'},404)
        except (ValueError,FileNotFoundError) as e:self.send({'error':str(e)},400)
    def do_POST(self):
        try:
            self.local()
            if urlsplit(self.path).path!='/api/analyse':self.send({'error':'Not found'},404);return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected application/json')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=21*1024*1024:raise ValueError('Upload exceeds 20 MiB')
            self.connection.settimeout(15);raw=self.rfile.read(size)
            if len(raw)!=size:raise ValueError('Incomplete request')
            d=json.loads(raw)
            if not isinstance(d,dict) or not isinstance(d.get('text'),str):raise ValueError('Expected file text')
            result=analyse(d['text'],d.get('kind'),d.get('cutoff'));result['csv']=csv_export(result);self.send(result)
        except (ValueError,TypeError,KeyError,OverflowError,RecursionError,OSError) as e:self.send({'error':str(e)},400)
def server(port=8782):return ThreadingHTTPServer(('127.0.0.1',port),Handler)
if __name__=='__main__':
    import torch;torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8782);args=p.parse_args();s=server(args.port)
    print(f'MotionReview: http://127.0.0.1:{s.server_port}',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
