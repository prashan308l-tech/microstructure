const EDGES=[2,4,6,8,12,16,24,32,48,64,96];
function patchFeats(px,W,x0,y0){ // px: RGBA array of 512x384 image
  const N=128,r=new Float64Array(N*N),gg=new Float64Array(N*N),b=new Float64Array(N*N),g=new Float64Array(N*N);
  for(let y=0;y<N;y++)for(let x=0;x<N;x++){const i=((y0+y)*W+x0+x)*4,k=y*N+x;
    r[k]=px[i];gg[k]=px[i+1];b[k]=px[i+2];g[k]=0.299*r[k]+0.587*gg[k]+0.114*b[k];}
  const f=[],n=N*N,gh=new Array(16).fill(0);let s=0,s2=0,sr=0,sg=0,sb=0,ss=0;
  for(let k=0;k<n;k++){gh[Math.min(Math.floor(g[k]/16),15)]++;s+=g[k];s2+=g[k]*g[k];sr+=r[k];sg+=gg[k];sb+=b[k];
    const mx=Math.max(r[k],gg[k],b[k]),mn=Math.min(r[k],gg[k],b[k]);ss+=(mx-mn)/(mx+1e-6);}
  for(const v of gh)f.push(v/n);
  const mean=s/n;f.push(mean/255,Math.sqrt(Math.max(s2/n-mean*mean,0))/255,sr/n/255,sg/n/255,sb/n/255,ss/n);
  const m=N-4,mh=new Array(12).fill(0);
  for(let y=2;y<N-2;y++)for(let x=2;x<N-2;x++){
    const gx=g[y*N+x+1]-g[y*N+x-1],gy=g[(y+1)*N+x]-g[(y-1)*N+x],mg=Math.sqrt(gx*gx+gy*gy);
    let bi=0;for(const e of EDGES)if(mg>=e)bi++;mh[bi]++;}
  for(const v of mh)f.push(v/(m*m));
  for(const rr of [1,2]){const h=new Array(9).fill(0);
    const O=[[-rr,-rr],[-rr,0],[-rr,rr],[0,-rr],[0,rr],[rr,-rr],[rr,0],[rr,rr]];
    for(let y=2;y<N-2;y++)for(let x=2;x<N-2;x++){let c=0;const cv=g[y*N+x];
      for(const [dy,dx] of O)if(g[(y+dy)*N+x+dx]>cv)c++;h[c]++;}
    for(const v of h)f.push(v/(m*m));}
  return f;
}
function predict(px,M){const P=new Array(M.classes.length).fill(0);let cnt=0;
  for(let y=0;y<384;y+=128)for(let x=0;x<512;x+=128){
    const f=patchFeats(px,512,x,y),z=M.W.map((w,i)=>w.reduce((a,v,j)=>a+v*(f[j]-M.mean[j])/M.scale[j],M.b[i]));
    const mx=Math.max(...z),e=z.map(v=>Math.exp(v-mx)),se=e.reduce((a,v)=>a+v,0);
    e.forEach((v,i)=>P[i]+=v/se);cnt++;}
  return P.map(v=>v/cnt);}
