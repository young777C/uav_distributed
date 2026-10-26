"""Static server + frame-sequence player for replay_videos/<seed>/NNNN.jpg (+ legacy <seed>.mp4). 127.0.0.1 only."""
import json, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
d, port = sys.argv[1], int(sys.argv[2])
os.chdir(d)

PAGE = """<!doctype html><meta charset=utf-8><title>Stress seed replay</title>
<style>body{background:#111;color:#ddd;font:13px system-ui;margin:12px}select,button{background:#222;color:#ddd;border:1px solid #444;padding:3px 8px}
#side{float:left;width:230px;height:92vh;overflow:auto}#side div{padding:2px 6px;cursor:pointer;border-bottom:1px solid #222}#side div:hover{background:#222}
#main{margin-left:245px}img{width:640px;height:640px;background:#000;display:block}.s{color:#888}</style>
<div id=side></div><div id=main>
<div>集合 <select id=fs><option value="">全部</option><option>devS</option><option>smoke</option><option>tune</option><option>confirm</option></select>
town <select id=ft><option value="">全部</option></select></div>
<h3 id=h>选择左侧种子</h3>
<img id=im><div><button id=pl>▶/⏸</button> <input id=sl type=range min=0 max=0 value=0 style="width:480px"> <span id=tv></span>
速度 <select id=sp><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></div>
<p class=s>画面 = 该种子场景(stress_v1)由专家飞行的 CARLA 重建,不是各方法闭环测试的录像。绿框=真目标,青框=其他车。</p></div>
<script>
const $=i=>document.getElementById(i);let L=[],cur=null,n=0,i=0,play=false,cache={};
const url=(s,k)=>'/'+s+'/'+String(k).padStart(4,'0')+'.jpg';
async function load(){L=await (await fetch('/api/list')).json();
 [...new Set(L.map(x=>x.town))].sort().forEach(t=>{if(![...$('ft').options].some(o=>o.value==t)){const o=document.createElement('option');o.textContent=t;$('ft').append(o)}});render()}
function render(){const fs=$('fs').value,ft=$('ft').value;$('side').innerHTML='';
 L.filter(x=>(!fs||x.set==fs)&&(!ft||x.town==ft)).forEach(x=>{const e=document.createElement('div');
  e.innerHTML=x.seed+' <span class=s>'+x.set+' '+x.town+(x.done?(x.skipped?' 跳过':' '+(x.frames/10).toFixed(0)+'s'):' ⏳')+'</span>';
  if(x.done&&!x.skipped)e.onclick=()=>open_(x);$('side').append(e)});}
function open_(x){cur=x;n=x.frames;i=0;cache={};$('h').textContent='seed '+x.seed+' · '+x.set+' · '+x.town;$('sl').max=n-1;show()}
function show(){if(!cur)return;$('im').src=url(cur.seed,i);$('sl').value=i;$('tv').textContent=(i/10).toFixed(1)+' / '+(n/10).toFixed(1)+' s';
 for(let k=i+1;k<i+40&&k<n;k++){if(!cache[k]){cache[k]=new Image();cache[k].src=url(cur.seed,k)}}}
$('sl').oninput=e=>{i=+e.target.value;show()};$('pl').onclick=()=>play=!play;$('fs').onchange=$('ft').onchange=render;
(function tick(){if(play&&cur){i=(i+1)%n;show()}setTimeout(tick,100/parseFloat($('sp').value))})();
load();setInterval(load,60000);
</script>"""

class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._s(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/list":
            meta = json.load(open("seeds_meta.json"))
            out = []
            for s, m in meta.items():
                f = os.path.join(s, "done.json")
                dj = json.load(open(f)) if os.path.exists(f) else None
                out.append({"seed": int(s), "set": m["set"], "town": m["town"], "done": bool(dj),
                            "frames": dj["frames"] if dj else 0, "skipped": bool(dj and dj.get("skipped"))})
            self._s(json.dumps(out).encode(), "application/json")
        else:
            super().do_GET()
    def _s(self, b, t):
        self.send_response(200); self.send_header("Content-Type", t); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
