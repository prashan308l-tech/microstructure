import numpy as np, json, subprocess
from PIL import Image
exec(open("train.py").read().split("meths=")[0])
im=Image.open(sorted((root/classes[1]).glob("*"))[0]).convert("RGB").resize((512,384),Image.BILINEAR)
a=np.asarray(im,dtype=float)
M=json.load(open("model.json"))
rgba=np.dstack([a,np.full(a.shape[:2],255)]).astype(np.uint8).tobytes()
open("px.bin","wb").write(rgba)
pf=np.array(feats(a[128:256,256:384]))
pp=np.zeros(3)
for y in range(0,384,128):
  for x in range(0,512,128):
    z=(np.array(feats(a[y:y+128,x:x+128]))-M["mean"])/M["scale"]
    l=np.array(M["W"])@z+M["b"];e=np.exp(l-l.max());pp+=e/e.sum()
print("py probs",(pp/12).round(4))
open("t.js","w").write(open("lib.js").read()+"""
const fs=require('fs'),px=fs.readFileSync('px.bin'),M=JSON.parse(fs.readFileSync('model.json'));
const f=patchFeats(px,512,256,128);console.log('f',JSON.stringify(f.slice(0,3)),f.length);
console.log('js probs',predict(px,M).map(v=>+v.toFixed(4)));
const ref=%s;let d=0;f.forEach((v,i)=>d=Math.max(d,Math.abs(v-ref[i])));console.log('max feat diff',d);
"""%json.dumps(pf.tolist()))
