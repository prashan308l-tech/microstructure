import numpy as np, json, random
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
root=Path("d/3_familiesOfSteels")
classes=sorted(p.name for p in root.iterdir() if p.is_dir())
EDGES=[0,2,4,6,8,12,16,24,32,48,64,96,1e9]
def feats(p):  # p: 128x128x3 float RGB 0-255
    g=0.299*p[...,0]+0.587*p[...,1]+0.114*p[...,2]
    f=[]
    gb=np.minimum(np.floor(g/16),15).astype(int)
    f+=list(np.bincount(gb.ravel(),minlength=16)/g.size)
    f+=[g.mean()/255,g.std()/255]
    f+=[p[...,0].mean()/255,p[...,1].mean()/255,p[...,2].mean()/255]
    mx=p.max(2);mn=p.min(2);f.append(((mx-mn)/(mx+1e-6)).mean())
    c=g[2:-2,2:-2]
    gx=g[2:-2,3:-1]-g[2:-2,1:-3]; gy=g[3:-1,2:-2]-g[1:-3,2:-2]
    m=np.sqrt(gx*2+gy**2)
    mb=np.zeros(m.shape,int)
    for e in EDGES[1:-1]: mb+= (m>=e)
    f+=list(np.bincount(mb.ravel(),minlength=12)/m.size)
    H,W=g.shape
    for r in (1,2):
        cnt=np.zeros(c.shape,int)
        for dy,dx in [(-r,-r),(-r,0),(-r,r),(0,-r),(0,r),(r,-r),(r,0),(r,r)]:
            cnt+= (g[2+dy:H-2+dy,2+dx:W-2+dx]>c)
        f+=list(np.bincount(cnt.ravel(),minlength=9)/cnt.size)
    return f
def patches(img,aug=False):
    a=np.asarray(img,dtype=float)
    if aug:
        a=np.clip(a*random.uniform(.9,1.1)+random.uniform(-12,12),0,255)
    return [feats(a[y:y+128,x:x+128]) for y in range(0,384,128) for x in range(0,512,128)]
meths=[Image.BILINEAR,Image.BICUBIC,Image.LANCZOS,Image.BOX]
imgs=[];labels=[]
for i,c in enumerate(classes):
    for f in sorted((root/c).glob("*")):
        if f.suffix.lower() in(".jpg",".jpeg",".png"):
            imgs.append(Image.open(f).convert("RGB"));labels.append(i)
labels=np.array(labels);print(classes,len(imgs))
def build(idx,aug):
    X=[];Y=[]
    for k in idx:
        for m in (meths if aug else [Image.BILINEAR]):
            ps=patches(imgs[k].resize((512,384),m),aug)
            X+=ps;Y+=[labels[k]]*len(ps)
    return np.array(X),np.array(Y)
accs=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(labels)),labels):
    X,Y=build(tr,True);sc=StandardScaler().fit(X)
    clf=LogisticRegression(C=0.3,max_iter=3000).fit(sc.transform(X),Y)
    ok=0
    for k in te:
        P=clf.predict_proba(sc.transform(np.array(patches(imgs[k].resize((512,384),Image.BICUBIC)))))
        ok+= P.mean(0).argmax()==labels[k]
    accs.append(ok/len(te));print("fold",accs[-1])
print("CV image accuracy",np.mean(accs))
X,Y=build(range(len(imgs)),True);sc=StandardScaler().fit(X)
clf=LogisticRegression(C=0.3,max_iter=3000).fit(sc.transform(X),Y)
json.dump({"classes":classes,"mean":sc.mean_.round(6).tolist(),"scale":sc.scale_.round(6).tolist(),
 "W":clf.coef_.round(5).tolist(),"b":clf.intercept_.round(5).tolist(),"cv":round(float(np.mean(accs)),3)},open("model.json","w"))
