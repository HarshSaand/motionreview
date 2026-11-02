"""Download only 32 named CMU captures; no crawling of the original database."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV='09a07f54f3bbb58797325f009282d0b2048a2871'
SPLITS={'train':{2:[1,2,3,4],7:[1,2,3,4],8:[1,2,3,4],9:[1,2,3,4],16:[11,12,35,36]},'validation':{35:[1,2,17,18]},'test':{38:[1,2,3,4],39:[1,2,3,4]}}
def main():
    records=[];out=ROOT/'data';out.mkdir(exist_ok=True)
    manifest=ROOT/'artifacts/data_manifest.json'
    expected={r['file']:r['sha256'] for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    for split,subjects in SPLITS.items():
        for subject,trials in subjects.items():
            for trial in trials:
                name=f'{subject:02}_{trial:02}.bvh';url=f'https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/{REV}/data/{subject:03}/{name}'
                p=out/name
                if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=40).read())
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                if name in expected and digest!=expected[name]:raise ValueError('Checksum mismatch: '+name)
                records.append(dict(file=name,split=split,session=subject,url=url,sha256=digest))
    (ROOT/'artifacts/data_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Downloaded {len(records)} real captures')
if __name__=='__main__':main()
