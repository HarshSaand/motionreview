"""Render saved real inference output as a shareable vector image."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=json.loads((ROOT/'artifacts/example_review.json').read_text());parts=[]
 for i,(label,key,color) in enumerate([('Last observed pose','history','#8397a3'),('MLP forecast at 0.50 s','forecast','#187d94'),('Actual recorded future','observed_future','#ca8235')]):
  pose=d[key][-1] if key!='forecast' else d[key]['mlp'][-1];x0=45+i*370
  parts.append(f'<rect x="{x0}" y="150" width="350" height="365" rx="14" fill="white"/><text x="{x0+20}" y="184" font-size="18" font-weight="bold">{label}</text>')
  points=[(x0+175+(p[0]*.906+p[2]*.423)*310,330-p[1]*310) for p in pose]
  for j,parent in enumerate(d['parents']):
   if parent>=0:
    a,b=points[j],points[parent];parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
  for x,y in points:parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
 score=d['observed_error_pct_body_scale'];svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650"><rect width="1200" height="650" fill="#edf3f5"/><g font-family="Arial,sans-serif" fill="#173140"><text x="45" y="48" font-size="14" fill="#187d94" letter-spacing="2">MOTIONREVIEW / REAL CAPTURE OUTPUT</text><text x="45" y="96" font-size="34" font-weight="bold">Inspect the motion before trusting the forecast.</text><text x="45" y="126" font-size="17" fill="#526b78">CMU capture 38_01 · 15 observed frames · 15 predicted frames · root-relative coordinates</text>{''.join(parts)}<text x="45" y="560" font-size="22">MLP error: {score['mlp']:.2f}% · last-pose error: {score['persistence']:.2f}% of observed body scale</text><text x="45" y="596" font-size="16" fill="#526b78">This single example illustrates the tool; the separate benchmark covers 8 held-out captures and 183 windows.</text><text x="45" y="622" font-size="14" fill="#526b78">Source: CMU Graphics Lab, NSF EIA-0196217 · BVH conversion: Bruce Hahne · values from example_review.json</text></g></svg>'''
 (ROOT/'artifacts/motion-result.svg').write_text(svg)
if __name__=='__main__':main()
