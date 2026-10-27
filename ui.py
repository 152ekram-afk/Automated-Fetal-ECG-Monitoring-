"""Visual helpers for the Streamlit app: animated CSS, hero banner, leaderboard and live monitor.

Everything here returns plain HTML/CSS/JS strings, so it can be tested without Streamlit.
All HTML passed to st.markdown must be on ONE line (no blank lines, no 4-space indents),
otherwise the markdown parser breaks it - hence _compact().
"""
import base64
import html as _html
import json
import math

import numpy as np

TEAL, DEEP, ROSE, INK = "#0f9d9a", "#0b3b4a", "#e0457b", "#1f2a37"


def _compact(s):
    return "".join(line.strip() for line in s.strip().splitlines())


# ----------------------------------------------------------------------------- CSS
_CSS = """
html{scroll-behavior:smooth}
@keyframes fecgFadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes fecgFadeIn{from{opacity:0}to{opacity:1}}
@keyframes fecgBeat{0%,100%{transform:scale(1)}15%{transform:scale(1.25)}30%{transform:scale(1)}45%{transform:scale(1.15)}60%{transform:scale(1)}}
@keyframes fecgGrow{from{width:0}}
@keyframes fecgShine{from{background-position:0% 50%}to{background-position:100% 50%}}
.block-container{animation:fecgFadeUp .55s cubic-bezier(.2,.8,.2,1) both}
[data-testid="stMetric"]{background:#fff;border:1px solid #e3e9f0;border-radius:16px;padding:14px 16px;box-shadow:0 1px 2px rgba(16,42,67,.05);transition:transform .25s cubic-bezier(.2,.8,.2,1),box-shadow .25s ease,border-color .25s ease;animation:fecgFadeUp .6s cubic-bezier(.2,.8,.2,1) both}
[data-testid="stMetric"]:hover{transform:translateY(-4px);box-shadow:0 10px 24px rgba(16,42,67,.12);border-color:rgba(15,157,154,.45)}
[data-testid="stHorizontalBlock"]>div:nth-child(2) [data-testid="stMetric"]{animation-delay:.07s}
[data-testid="stHorizontalBlock"]>div:nth-child(3) [data-testid="stMetric"]{animation-delay:.14s}
[data-testid="stHorizontalBlock"]>div:nth-child(4) [data-testid="stMetric"]{animation-delay:.21s}
[data-testid="stHorizontalBlock"]>div:nth-child(5) [data-testid="stMetric"]{animation-delay:.28s}
button[data-baseweb="tab"]{transition:color .25s ease,background-color .25s ease,transform .2s ease}
button[data-baseweb="tab"]:hover{transform:translateY(-1px)}
[data-baseweb="tab-highlight"]{transition:all .35s cubic-bezier(.4,0,.2,1)!important}
div[data-baseweb="tab-panel"]{animation:fecgFadeUp .45s cubic-bezier(.2,.8,.2,1) both}
.stButton>button,.stDownloadButton>button{border-radius:12px;transition:transform .2s ease,box-shadow .2s ease,background-color .2s ease}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(16,42,67,.15)}
[data-testid="stAlert"]{border-radius:14px;animation:fecgFadeIn .5s ease both}
[data-testid="stImage"] img{animation:fecgFadeIn .7s ease both;border-radius:12px}
[data-testid="stDataFrame"]{animation:fecgFadeIn .6s ease both}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f3f8fb 0%,#e9f1f6 100%)}
.fecg-hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 32px;margin:2px 0 16px;color:#fff;background:linear-gradient(120deg,#0b3b4a,#0f9d9a 55%,#e0457b);background-size:200% 200%;animation:fecgShine 9s ease-in-out infinite alternate,fecgFadeUp .7s cubic-bezier(.2,.8,.2,1) both;box-shadow:0 14px 34px rgba(11,59,74,.25)}
.fecg-hero h1{margin:0 0 8px;padding:0;font-size:2rem;line-height:1.15;color:#fff;position:relative}
.fecg-hero p{margin:0;max-width:54%;opacity:.93;font-size:1rem;line-height:1.45;color:#fff;position:relative}
.fecg-hero .heart{display:inline-block;transform-origin:center;animation:fecgBeat 1.3s ease-in-out infinite}
.fecg-hero img.ecg{position:absolute;right:0;bottom:0;width:44%;height:100%;opacity:.5;pointer-events:none;object-fit:cover}
.lb{margin:8px 0 4px}
.lb-row{display:grid;grid-template-columns:minmax(130px,230px) 1fr auto;gap:14px;align-items:center;padding:10px 14px;margin:7px 0;border-radius:14px;background:#fff;border:1px solid #e3e9f0;animation:fecgFadeUp .55s cubic-bezier(.2,.8,.2,1) both;transition:transform .2s ease,box-shadow .2s ease}
.lb-row:hover{transform:translateX(5px);box-shadow:0 8px 20px rgba(16,42,67,.10)}
.lb-row.ours{border-color:rgba(224,69,123,.55);background:linear-gradient(90deg,#fff4f8,#fff)}
.lb-name{font-weight:600;font-size:.93rem;color:#1f2a37;line-height:1.25}
.lb-sub{display:block;font-weight:400;font-size:.73rem;color:#64748b;margin-top:2px}
.lb-bar{height:12px;background:#eef2f6;border-radius:999px;overflow:hidden}
.lb-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#0f9d9a,#3ddc97);animation:fecgGrow 1s cubic-bezier(.2,.8,.2,1) both}
.lb-row.ours .lb-fill{background:linear-gradient(90deg,#e0457b,#ff8fab)}
.lb-row.base .lb-fill{background:linear-gradient(90deg,#94a3b8,#cbd5e1)}
.lb-val{font-variant-numeric:tabular-nums;font-weight:700;color:#0b3b4a;min-width:3.2em;text-align:right}
@media (max-width:640px){.fecg-hero p{max-width:100%}.fecg-hero{padding:22px 20px}.fecg-hero h1{font-size:1.45rem}.fecg-hero img.ecg{width:100%;opacity:.3}.lb-row{grid-template-columns:1fr auto}.lb-bar{grid-column:1/-1;order:3}}
"""

_NO_MOTION = ("*,*::before,*::after{animation:none!important;transition:none!important;"
              "scroll-behavior:auto!important}")


def css(animations=True):
    """The <style> block. With animations=False (or when the OS asks for reduced motion) all motion is off."""
    out = _CSS + "@media (prefers-reduced-motion:reduce){" + _NO_MOTION + "}"
    if not animations:
        out += _NO_MOTION
    return "<style>" + out.strip() + "</style>"


# ----------------------------------------------------------------------------- hero
_ECG_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 120" preserveAspectRatio="none">'
    '<style>.t{fill:none;stroke:#fff;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;'
    'stroke-dasharray:1000;stroke-dashoffset:1000;animation:d 3.6s linear infinite}'
    '@keyframes d{0%{stroke-dashoffset:1000;opacity:1}70%{stroke-dashoffset:0;opacity:1}'
    '100%{stroke-dashoffset:0;opacity:0}}</style>'
    '<path class="t" pathLength="1000" d="M0,62 L70,62 L88,62 L100,30 L112,96 L124,48 L136,62 '
    'L250,62 L268,62 L280,22 L294,102 L306,50 L318,62 L430,62 L448,62 L460,28 L472,98 '
    'L484,52 L496,62 L600,62"/></svg>')


def hero_html(title="Automated Fetal ECG Monitoring",
              subtitle="Separates the baby's heartbeat from the mother's using abdominal ECG, "
                       "then estimates the fetal heart rate with machine learning."):
    b64 = base64.b64encode(_ECG_SVG.encode()).decode()
    return _compact(f"""
    <div class="fecg-hero">
      <img class="ecg" alt="" src="data:image/svg+xml;base64,{b64}"/>
      <h1><span class="heart">&#10084;&#65039;</span> {_html.escape(title)}</h1>
      <p>{_html.escape(subtitle)}</p>
    </div>""")


# ----------------------------------------------------------------------------- leaderboard
def _num(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def leaderboard_html(df, metric="F1", ours_key="random_forest"):
    """Animated horizontal bars, one row per method, sorted by `metric` (a column of the summary table)."""
    d = df.copy()
    d[metric] = d[metric].astype(float)
    d = d.sort_values(metric, ascending=False, na_position="last")
    rows = []
    for i, (_, r) in enumerate(d.iterrows()):
        v = r[metric]
        pct = 0.0 if v is None or math.isnan(v) else max(0.0, min(1.0, v)) * 100
        cls = "lb-row"
        if r.get("Key") == ours_key:
            cls += " ours"
        elif r.get("Type") == "baseline":
            cls += " base"
        sub = f"Sens {_num(r.get('Sensitivity'), 2)} &middot; Prec {_num(r.get('Precision'), 2)}"
        if not math.isnan(float(r.get("AUC", float("nan")))):
            sub += f" &middot; AUC {_num(r.get('AUC'), 2)}"
        rows.append(
            f'<div class="{cls}" style="animation-delay:{i * 0.06:.2f}s">'
            f'<div class="lb-name">{_html.escape(str(r["Model"]))}<span class="lb-sub">{sub}</span></div>'
            f'<div class="lb-bar"><div class="lb-fill" style="width:{pct:.1f}%;animation-delay:{0.15 + i * 0.06:.2f}s"></div></div>'
            f'<div class="lb-val">{_num(v)}</div></div>')
    return '<div class="lb">' + "".join(rows) + "</div>"


# ----------------------------------------------------------------------------- live monitor
def live_payload(signal, fs, beats, ref=None, max_seconds=60, target_fs=250, win=6.0, label=""):
    """Down-sample the signal for the browser and collect beat times (seconds)."""
    step = max(int(round(fs / target_fs)), 1)
    n_keep = int(min(len(signal), max_seconds * fs))
    x = np.asarray(signal[:n_keep], dtype=float)[::step]
    ymax = float(np.percentile(np.abs(x), 99.5)) * 1.3 if len(x) else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    x = np.clip(x, -ymax, ymax)

    def secs(b):
        b = np.asarray(b if b is not None else [], dtype=float)
        b = b[b < n_keep]
        return [round(float(v) / fs, 3) for v in np.sort(b)]

    return {"fs": fs / step, "sig": [round(float(v), 2) for v in x], "beats": secs(beats),
            "ref": secs(ref), "dur": round(n_keep / fs, 3), "win": win,
            "ymax": round(ymax, 3), "label": label}


_MONITOR_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:transparent}
#wrap{background:#0b1620;border-radius:18px;padding:14px 16px 12px;color:#d7f5e9;box-shadow:0 10px 28px rgba(11,22,32,.35)}
#top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px}
#fhr{display:flex;align-items:baseline;gap:8px}
#heart{display:inline-block;color:#ff5d8f;font-size:26px;transform-origin:center}
#heart.pulse{animation:beat .5s ease-out}
@keyframes beat{0%{transform:scale(1)}30%{transform:scale(1.45)}100%{transform:scale(1)}}
#bpm{font-size:46px;font-weight:700;color:#3ddc97;font-variant-numeric:tabular-nums;line-height:1;min-width:2.3em;transition:color .3s}
#unit{font-size:13px;color:#7fa99a}
#meta{font-size:12px;color:#7fa99a;text-align:right;line-height:1.5}
canvas{display:block;width:100%;height:250px;border-radius:12px;background:#08121a}
#ctl{display:flex;align-items:center;gap:10px;margin-top:10px;flex-wrap:wrap}
button,select{font:inherit;font-size:13px;color:#d7f5e9;background:#143040;border:1px solid #1f4a5e;border-radius:10px;padding:6px 12px;cursor:pointer;transition:background .2s,transform .15s}
button:hover,select:hover{background:#1b4358}
button:active{transform:scale(.96)}
#seek{flex:1;min-width:120px;accent-color:#3ddc97}
#clock{font-size:12px;color:#7fa99a;font-variant-numeric:tabular-nums;min-width:4.5em;text-align:right}
@media (prefers-reduced-motion:reduce){#heart.pulse{animation:none}}
</style></head><body>
<div id="wrap">
  <div id="top">
    <div id="fhr"><span id="heart">&#9829;</span><span id="bpm">--</span><span id="unit">BPM</span></div>
    <div id="meta"><div id="lbl"></div><div id="count"></div></div>
  </div>
  <canvas id="cv"></canvas>
  <div id="ctl">
    <button id="play">Pause</button><button id="restart">Restart</button>
    <input id="seek" type="range" min="0" max="1000" value="0">
    <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select>
    <span id="clock">0.0 s</span>
  </div>
</div>
<script>
(function(){
const D = __DATA__;
const AUTOPLAY = __AUTOPLAY__;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const bpmEl=document.getElementById('bpm'), heart=document.getElementById('heart');
const clock=document.getElementById('clock'), countEl=document.getElementById('count');
const playBtn=document.getElementById('play'), seek=document.getElementById('seek'), speedSel=document.getElementById('speed');
document.getElementById('lbl').textContent = D.label || '';
let W=300,H=250,dpr=window.devicePixelRatio||1;
function resize(){const r=cv.getBoundingClientRect(); W=Math.max(r.width,200); H=Math.max(r.height,120); cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); ctx.setTransform(dpr,0,0,dpr,0,0);}
window.addEventListener('resize',resize); resize();
function ub(a,x){let lo=0,hi=a.length;while(lo<hi){const m=(lo+hi)>>1; if(a[m]<=x) lo=m+1; else hi=m;} return lo;}
function bpmAt(T){const k=ub(D.beats,T); if(k<2) return null; const v=[]; for(let i=Math.max(1,k-6);i<k;i++){const b=60/(D.beats[i]-D.beats[i-1]); if(b>=80&&b<=220) v.push(b);} if(!v.length) return null; v.sort((a,b)=>a-b); return v[v.length>>1];}
let T=AUTOPLAY?0:Math.min(D.win,D.dur), playing=AUTOPLAY, speed=1, prev=performance.now(), lastK=ub(D.beats,T), drag=false;
function pulse(){heart.classList.remove('pulse'); void heart.offsetWidth; heart.classList.add('pulse');}
function draw(){
  ctx.clearRect(0,0,W,H);
  const t1=T, t0=T-D.win, x_of=t=>(t-t0)/D.win*W;
  ctx.lineWidth=1;
  for(let g=Math.ceil(t0*2)/2; g<=t1; g+=0.5){const major=Math.abs(g-Math.round(g))<1e-6; ctx.strokeStyle=major?'rgba(80,160,130,.30)':'rgba(80,160,130,.12)'; const x=x_of(g); ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();}
  for(let j=1;j<5;j++){ctx.strokeStyle='rgba(80,160,130,.12)'; ctx.beginPath(); ctx.moveTo(0,H*j/5); ctx.lineTo(W,H*j/5); ctx.stroke();}
  const i0=Math.max(0,Math.floor(t0*D.fs)), i1=Math.min(D.sig.length-1,Math.floor(t1*D.fs));
  ctx.beginPath(); ctx.strokeStyle='#3ddc97'; ctx.lineWidth=1.6; ctx.lineJoin='round';
  for(let i=i0;i<=i1;i++){const x=x_of(i/D.fs), y=H/2-(D.sig[i]/D.ymax)*(H*0.42); if(i===i0) ctx.moveTo(x,y); else ctx.lineTo(x,y);}
  ctx.stroke();
  for(let i=ub(D.beats,t0-0.5);i<D.beats.length&&D.beats[i]<=t1;i++){
    const b=D.beats[i], x=x_of(b), age=t1-b;
    ctx.strokeStyle='rgba(255,93,143,.35)'; ctx.lineWidth=1; ctx.setLineDash([3,4]); ctx.beginPath(); ctx.moveTo(x,16); ctx.lineTo(x,H); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#ff5d8f'; ctx.beginPath(); ctx.moveTo(x-5,4); ctx.lineTo(x+5,4); ctx.lineTo(x,15); ctx.closePath(); ctx.fill();
    if(age>=0&&age<0.5){const k=age/0.5; ctx.strokeStyle='rgba(255,93,143,'+(0.8*(1-k)).toFixed(3)+')'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(x,H/2,8+k*26,0,6.2832); ctx.stroke();}
  }
  for(let i=ub(D.ref,t0-0.5);i<D.ref.length&&D.ref[i]<=t1;i++){const x=x_of(D.ref[i]); ctx.strokeStyle='#7fd1ff'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(x,H-12,5,0,6.2832); ctx.stroke();}
  ctx.fillStyle='rgba(215,245,233,.55)'; ctx.font='11px system-ui,sans-serif';
  ctx.fillText('\\u25BC detected fetal beat'+(D.ref.length?'   \\u25EF reference (Direct_1)':''),10,H-30);
}
function ui(){
  const b=bpmAt(T); bpmEl.textContent=b?Math.round(b):'--';
  clock.textContent=T.toFixed(1)+' s'; countEl.textContent=ub(D.beats,T)+' beats detected';
  if(!drag) seek.value=Math.round(T/D.dur*1000);
}
function frame(now){
  const dt=Math.min((now-prev)/1000,0.1); prev=now;
  if(playing){T+=dt*speed; if(T>D.dur){T=0; lastK=0;}}
  const k=ub(D.beats,T); if(k>lastK&&playing) pulse(); lastK=k;
  draw(); ui(); requestAnimationFrame(frame);
}
playBtn.onclick=()=>{playing=!playing; playBtn.textContent=playing?'Pause':'Play'; prev=performance.now();};
document.getElementById('restart').onclick=()=>{T=0; lastK=0; playing=true; playBtn.textContent='Pause';};
speedSel.onchange=()=>{speed=parseFloat(speedSel.value);};
seek.oninput=()=>{drag=true; T=seek.value/1000*D.dur; lastK=ub(D.beats,T);};
seek.onchange=()=>{drag=false;};
playBtn.textContent=playing?'Pause':'Play';
requestAnimationFrame(frame);
})();
</script></body></html>"""


def live_monitor_html(payload, autoplay=True):
    data = json.dumps(payload, separators=(",", ":"))
    return (_MONITOR_HTML.replace("__DATA__", data)
            .replace("__AUTOPLAY__", "true" if autoplay else "false"))


# ----------------------------------------------------------------------------- Streamlit glue
def show_frame(html, height=400):
    """Render a self-contained HTML document in an iframe, whichever API this Streamlit version has."""
    import streamlit as st
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=height)
            return
        except Exception:
            pass
    import streamlit.components.v1 as components
    components.html(html, height=height)
