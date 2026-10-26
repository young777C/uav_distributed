"""1026 viewer for scripts/scene_probe.py recordings: frame player + trajectory mini-map + per-seed / per-town scene
statistics next to the stress_v1 reference (v1_reference.json in the record dir). Binds 127.0.0.1 only.
  python3 scene_probe_index.py <record_dir> 8773"""
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

d, port = sys.argv[1], int(sys.argv[2])
os.chdir(d)

PAGE = r"""<!doctype html><meta charset=utf-8><title>Stress v2 scene probe</title>
<style>
body{background:#111;color:#ddd;font:13px system-ui;margin:10px}select,button{background:#222;color:#ddd;border:1px solid #444;padding:3px 8px}
#side{float:left;width:300px;height:94vh;overflow:auto}#side .g{color:#9cf;margin-top:8px;font-weight:600}
#side .r{padding:2px 6px;cursor:pointer;border-bottom:1px solid #222}#side .r:hover,#side .r.on{background:#223}
#main{margin-left:315px}.s{color:#888}table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #333;padding:2px 6px;text-align:right}
th{background:#1b1b1b}td:first-child,th:first-child{text-align:left}.up{color:#7d7}.dn{color:#e77}
#row{display:flex;gap:10px;align-items:flex-start}img{width:512px;height:512px;background:#000}canvas{background:#0b0b0b;border:1px solid #333}
.lg span{display:inline-block;width:10px;height:10px;margin:0 3px 0 10px}
</style>
<div id=side></div><div id=main>
<details open><summary>按 town 汇总(v2 均值 / v1 参考,v1 = stress_v1 专家录制 177 局)</summary><div id=agg></div></details>
<h3 id=h>选择左侧种子</h3>
<div id=row><div><img id=im><div><button id=pl>▶/⏸</button> <input id=sl type=range min=0 max=0 value=0 style="width:330px"> <span id=tv></span>
速度 <select id=sp><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></div></div>
<div><canvas id=mm width=420 height=420></canvas>
<div class=lg><span style="background:#555"></span>规划路线<span style="background:#3c3"></span>目标可见<span style="background:#e33"></span>画面内被遮挡<span style="background:#e90"></span>出画<span style="background:#a6f"></span>E3 窗口</div>
<div id=st></div></div></div>
<p class=s>画面 = 该种子场景(stress_v2)由专家飞行重建,不是任何方法的闭环录像。框:绿=目标可见,红=目标在画面内被遮挡,黄=克隆车(粗=可见),灰=其他干扰车。顶栏 OCCLUDED/OUT/E3 = 当前帧状态。</p>
</div>
<script>
const $=i=>document.getElementById(i);let L=[],REF={},cur=null,F=[],n=0,i=0,play=false,cache={};
const KEYS=[['still_frac','目标静止%',100,-1],['stops_2s','停车≥2s/局',1,-1],['turns_45','转弯≥45°/局',1,1],['speed_moving','行驶速度',1,1],
 ['occ_frac_inframe','画面内遮挡%',100,1],['occ_runs_1s','遮挡≥1s/局',1,1],['occ_runs_2s','遮挡≥2s/局',1,1],['lost_runs_2s','不可见≥2s/局',1,1],
 ['clone_still_frac','克隆静止%',100,-1],['other_still_frac','干扰车静止%',100,-1],['clone_covis_frac','目标可见时克隆同屏%',100,1]];
const url=(s,k)=>'/'+s+'/'+String(k).padStart(4,'0')+'.jpg';
const f=(v,m)=>v==null?'–':(v*m).toFixed(m==100?0:2);
async function load(){[L,REF]=await Promise.all([(await fetch('/api/list')).json(),(await fetch('/v1_reference.json')).json().catch(()=>({}))]);side();agg()}
function side(){const s=$('side');s.innerHTML='';const by={};L.forEach(x=>{(by[(x.tier||'-')+' · '+x.town]=by[(x.tier||'-')+' · '+x.town]||[]).push(x)});
 Object.keys(by).sort().forEach(g=>{const e=document.createElement('div');e.className='g';e.textContent=g;s.append(e);
  by[g].forEach(x=>{const r=document.createElement('div');r.className='r'+(cur&&cur.seed==x.seed?' on':'');const st=x.stats||{};
   r.innerHTML=x.seed+' <span class=s>'+(x.done?(x.skipped?'跳过 '+(x.reason||''):('静止'+f(st.still_frac,100)+'% 转'+(st.turns_45??'–')+' 遮≥2s '+(st.occ_runs_2s??'–'))):'⏳')+'</span>';
   if(x.done&&!x.skipped)r.onclick=()=>open_(x);s.append(r)})})}
function agg(){const T={};L.filter(x=>x.done&&!x.skipped&&x.stats).forEach(x=>{(T[x.town]=T[x.town]||{tier:x.tier,v:[]}).v.push(x.stats)});
 let h='<table><tr><th>town</th><th>档</th><th>n</th>'+KEYS.map(k=>'<th>'+k[1]+'</th>').join('')+'</tr>';
 Object.keys(T).sort().forEach(t=>{const v=T[t].v,r=REF[t]||{};h+='<tr><td>'+t+'</td><td>'+(T[t].tier||'')+'</td><td>'+v.length+'</td>'+KEYS.map(k=>{
   const a=v.map(s=>s[k[0]]).filter(z=>z!=null);if(!a.length)return '<td>–</td>';const m=a.reduce((p,q)=>p+q,0)/a.length,b=r[k[0]];
   const cls=(b==null||k[0]=='clone_covis_frac')?'':((m-b)*k[3]>0?'up':'dn');
   return '<td class='+cls+'>'+f(m,k[2])+(b!=null&&k[0]!='clone_covis_frac'?' <span class=s>/ '+f(b,k[2])+'</span>':'')+'</td>'}).join('')+'</tr>'});
 $('agg').innerHTML=h+'</table><span class=s>绿=朝"更难"方向变化,红=反向;v1 无 Town03 数据。</span>'}
async function open_(x){cur=x;n=x.frames;i=0;cache={};side();$('h').textContent='seed '+x.seed+' · '+x.town+' · '+(x.tier||'');$('sl').max=n-1;
 F=(await (await fetch('/api/frames/'+x.seed)).json());stats();show()}
function stats(){const st=cur.stats||{},r=cur.route||{};let h='<table>'+KEYS.map(k=>'<tr><td>'+k[1]+'</td><td>'+f(st[k[0]],k[2])+'</td></tr>').join('');
 h+='<tr><td>unjam 迁移 / 换道</td><td>'+(st.unjam_relocated??'–')+' / '+(st.unjam_lane_kicks??'–')+'</td></tr><tr><td>克隆 / 干扰车</td><td>'+cur.n_clones+' / '+cur.n_distractors+'</td></tr>';
 h+='<tr><td>规划:长度 / 转弯</td><td>'+(r.length_m??'–')+' m / '+((r.turns||[]).length)+'</td></tr><tr><td>规划遮挡段(入口m,跨度m)</td><td>'+((r.occluders||[]).map(o=>o.enter_m.toFixed(0)+','+o.span_m.toFixed(0)+' '+o.label).join('<br>')||'无')+'</td></tr></table>';$('st').innerHTML=h}
function mm(){const c=$('mm'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);if(!F.length)return;
 const P=(cur.polyline||[]).concat(F.map(r=>r.tp));let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;P.forEach(p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1])});
 const sc=380/Math.max(x1-x0,y1-y0,1),X=p=>20+(p[0]-x0)*sc,Y=p=>20+(p[1]-y0)*sc;
 g.strokeStyle='#555';g.lineWidth=6;g.beginPath();(cur.polyline||[]).forEach((p,k)=>k?g.lineTo(X(p),Y(p)):g.moveTo(X(p),Y(p)));g.stroke();
 g.lineWidth=3;for(let k=1;k<F.length;k++){const r=F[k];g.strokeStyle=r.e3?'#a6f':(!r.tin?'#e90':(r.tv?'#3c3':'#e33'));g.beginPath();g.moveTo(X(F[k-1].tp),Y(F[k-1].tp));g.lineTo(X(r.tp),Y(r.tp));g.stroke()}
 const r=F[Math.min(i,F.length-1)];g.fillStyle='#fff';g.beginPath();g.arc(X(r.tp),Y(r.tp),5,0,7);g.fill();
 (r.cl||[]).forEach(q=>{g.fillStyle='#fd0';g.fillRect(X(q)-3,Y(q)-3,6,6)})}
function show(){if(!cur)return;$('im').src=url(cur.seed,i);$('sl').value=i;$('tv').textContent=(i/10).toFixed(1)+' / '+(n/10).toFixed(1)+' s';mm();
 for(let k=i+1;k<i+40&&k<n;k++){if(!cache[k]){cache[k]=new Image();cache[k].src=url(cur.seed,k)}}}
$('sl').oninput=e=>{i=+e.target.value;show()};$('pl').onclick=()=>play=!play;
(function tick(){if(play&&cur){i=(i+1)%n;show()}setTimeout(tick,100/parseFloat($('sp').value))})();
load();setInterval(async()=>{L=await (await fetch('/api/list')).json();side();agg()},60000);
</script>"""


class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._s(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/list":
            meta = json.load(open("seeds_meta.json")) if os.path.exists("seeds_meta.json") else {}
            out = []
            for s, m in meta.items():
                f = os.path.join(s, "done.json")
                dj = json.load(open(f)) if os.path.exists(f) else None
                row = {"seed": int(s), "set": m.get("set"), "town": m["town"], "done": bool(dj),
                       "frames": dj["frames"] if dj else 0, "skipped": bool(dj and dj.get("skipped"))}
                if dj:
                    row.update({k: dj.get(k) for k in ("tier", "reason", "stats", "route", "polyline", "n_clones",
                                                       "n_distractors")})
                out.append(row)
            self._s(json.dumps(sorted(out, key=lambda r: r["seed"])).encode(), "application/json")
        elif self.path.startswith("/api/frames/"):
            s = os.path.basename(self.path)
            p = os.path.join(s, "frames.jsonl")
            rows = []
            if s.isdigit() and os.path.exists(p):
                for line in open(p):
                    r = json.loads(line)
                    rows.append({"tp": r["tp"][:2], "tv": r["tv"], "tin": r["tin"], "e3": r["e3"],
                                 "cl": [c[:2] for c in r["cl"]]})
            self._s(json.dumps(rows).encode(), "application/json")
        else:
            super().do_GET()

    def _s(self, b, t):
        self.send_response(200)
        self.send_header("Content-Type", t)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
