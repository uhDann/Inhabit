"""Build a 2AFC ("which is the real photo?") human study from the eval panels.

Splits each eval cmp_*.png (real | render) into the two halves, randomises left/right,
and writes a self-contained HTML page. The viewer clicks the one they think is the real
photo; the page tallies discrimination accuracy. ~50% = indistinguishable (the goal);
>65-70% = people can still tell. No server needed -- open the HTML locally.
"""
from __future__ import annotations
import argparse, glob, os, base64, json, random


def b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="runs/photoreal/eval")
    ap.add_argument("--out", default="runs/photoreal/twoafc.html")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    from PIL import Image
    import io
    panels = sorted(glob.glob(f"{a.eval_dir}/cmp_*.png"))[: a.n]
    trials = []
    for p in panels:
        im = Image.open(p); W, H = im.size; half = W // 2
        real = im.crop((0, 0, half, H)); rend = im.crop((half, 0, W, H))
        real_first = random.random() < 0.5
        def enc(x):
            buf = io.BytesIO(); x.save(buf, "PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        left, right = (real, rend) if real_first else (rend, real)
        trials.append({"left": enc(left), "right": enc(right),
                       "real": "left" if real_first else "right"})
    html = f"""<!doctype html><meta charset=utf-8><title>Real vs reconstruction (2AFC)</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;text-align:center}}
img{{max-width:46vw;border:2px solid #333;cursor:pointer;border-radius:8px}}
img:hover{{border-color:#1f6feb}} #s{{font-size:20px;margin:14px}}</style>
<h2>Click the image you think is the REAL photo</h2><div id=s></div>
<div><img id=L><img id=R></div>
<script>
const T={json.dumps(trials)};let i=0,correct=0;
function show(){{if(i>=T.length){{document.body.innerHTML=
 '<h2>Done</h2><p>Discrimination accuracy: '+(100*correct/T.length).toFixed(0)+'%'+
 ' (n='+T.length+'). ~50% = indistinguishable from real.</p>';return;}}
 L.src=T[i].left;R.src=T[i].right;
 document.getElementById('s').textContent='Trial '+(i+1)+' / '+T.length;}}
function pick(side){{if(side===T[i].real)correct++;i++;show();}}
L.onclick=()=>pick('left');R.onclick=()=>pick('right');show();
</script>"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(html)
    print("wrote", a.out, f"({len(trials)} trials)")


if __name__ == "__main__":
    main()
