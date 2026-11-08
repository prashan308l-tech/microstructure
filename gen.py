import json
M=json.load(open("model.json"))
M["names"]=["Austenitic Stainless Steel","Carbon Steel","Duplex Stainless Steel"]
lib=open("lib.js").read()
html='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Microstructure Classifier</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#14181f;--mu:#5d6676;--bd:#dde1e8;--ac:#2563eb;--ok:#16a34a;--wr:#d97706;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1218;--card:#181c24;--tx:#e8ebf0;--mu:#9aa3b2;--bd:#2a303b;--ac:#5b8cff;--ok:#4ade80;--wr:#fbbf24}}
:root[data-theme="dark"]{--bg:#0f1218;--card:#181c24;--tx:#e8ebf0;--mu:#9aa3b2;--bd:#2a303b;--ac:#5b8cff;--ok:#4ade80;--wr:#fbbf24}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.6rem;margin:0 0 4px}p.sub{color:var(--mu);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:18px;margin-bottom:16px}
#drop{border:2px dashed var(--bd);border-radius:14px;padding:36px 16px;text-align:center;cursor:pointer;color:var(--mu)}
#drop.over{border-color:var(--ac);color:var(--ac)}
#drop b{color:var(--tx)}
img#pv{max-width:100%;border-radius:10px;display:block;margin:0 auto 14px;max-height:340px}
.res{display:none}.top{font-size:1.4rem;font-weight:700;margin:0}
.badge{display:inline-block;font-size:.8rem;padding:2px 10px;border-radius:99px;border:1px solid currentColor;margin-left:8px;vertical-align:middle}
.row{margin:12px 0}.row .l{display:flex;justify-content:space-between;font-size:.92rem}
.bar{height:10px;background:var(--bd);border-radius:6px;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:var(--ac);width:0;transition:width .4s}
.note{font-size:.85rem;color:var(--mu)}.warn{color:var(--wr);font-weight:600}
button{background:var(--ac);color:#fff;border:0;border-radius:10px;padding:10px 16px;font:inherit;cursor:pointer}
</style></head><body><main>
<h1>Microstructure Classifier</h1>
<p class="sub">Upload a light-optical micrograph of steel and the model predicts its family. It runs entirely in your browser; images are not uploaded anywhere.</p>
<div class="card"><div id="drop"><b>Click to choose an image</b> or drag and drop it here<br><span class="note">JPG or PNG, ideally a 4:3 micrograph</span></div>
<input id="file" type="file" accept="image/*" hidden></div>
<div class="card res" id="res"><img id="pv" alt="Uploaded micrograph">
<p class="top"><span id="name"></span><span class="badge" id="badge"></span></p>
<div id="bars"></div><p class="note" id="msg"></p></div>
<div class="card note"><b>About this model.</b> Trained on 90 micrographs (30 each) of austenitic stainless, carbon and duplex stainless steel from the CRTI Algeria open dataset. Features: grey-level, colour, gradient and local-texture statistics over 128&times;128 patches, combined by logistic regression. Cross-validated image accuracy (held-out images): <b id="cv"></b>. It can only choose among these three families, so images of other materials, etchants or microscopes may be classified wrongly. Treat results as indicative, not as a lab verdict.</div>
</main>
<script>
const M=__MODEL__;
__LIB__
const $=id=>document.getElementById(id),drop=$("drop"),file=$("file");
$("cv").textContent=(M.cv*100).toFixed(1)+"%";
drop.onclick=()=>file.click();
["dragover","dragenter"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add("over")}));
["dragleave","drop"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove("over")}));
drop.addEventListener("drop",ev=>{if(ev.dataTransfer.files[0])run(ev.dataTransfer.files[0])});
file.onchange=()=>{if(file.files[0])run(file.files[0])};
function run(f){const url=URL.createObjectURL(f),im=new Image();
 im.onload=()=>{const c=document.createElement("canvas");c.width=512;c.height=384;const x=c.getContext("2d");
  x.imageSmoothingQuality="high";x.drawImage(im,0,0,512,384);
  const P=predict(x.getImageData(0,0,512,384).data,M),k=P.indexOf(Math.max(...P));
  $("pv").src=url;$("res").style.display="block";$("name").textContent=M.names[k];
  const p=P[k],b=$("badge");b.textContent=(p*100).toFixed(1)+"% confidence";b.style.color=p>=.7?"var(--ok)":"var(--wr)";
  $("bars").innerHTML=M.names.map((n,i)=>`<div class="row"><div class="l"><span>${n}</span><span>${(P[i]*100).toFixed(1)}%</span></div><div class="bar"><i style="width:${P[i]*100}%"></i></div></div>`).join("");
  $("msg").innerHTML=p<.7?'<span class="warn">Low confidence.</span> This image may differ from the training data (material, etchant, magnification); do not rely on this result.':"Prediction is the average over 12 patches of the image.";
  $("res").scrollIntoView({behavior:"smooth"});};
 im.onerror=()=>alert("Could not read this image.");im.src=url;}
</script></body></html>'''
html=html.replace("__MODEL__",json.dumps(M,separators=(",",":"))).replace("__LIB__",lib)
open("microstructure-classifier.html","w").write(html)
print(len(html)//1024,"KB")
