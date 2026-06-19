import json, os
D=os.path.join(os.path.dirname(__file__),'results')
runs=['stock_s42','stock_s43','stock_s44','bitnet_s42','bitnet_s43','bitnet_s44']
res={}
for r in runs:
    tl=el=ep=None
    for line in open(os.path.join(D,f'{r}_curves.jsonl')):
        try:o=json.loads(line)
        except:continue
        en=o.get('event_name');v=o.get('value')
        if en=='train.loss':tl=v
        elif en=='eval.loss':el=v
        elif en=='eval.perplexity':ep=v
    res[r]=(tl,el,ep)
print(f"{'run':12}{'train':>9}{'val':>9}{'val_ppl':>9}")
for r in runs:
    t,e,p=res[r];print(f"{r:12}{t:9.4f}{e:9.4f}{p:9.4f}")
def st(v):
    m=sum(v)/len(v);return m,(sum((x-m)**2 for x in v)/len(v))**0.5
for g in['stock','bitnet']:
    tr=[res[f'{g}_s{s}'][0] for s in(42,43,44)];va=[res[f'{g}_s{s}'][1] for s in(42,43,44)];pp=[res[f'{g}_s{s}'][2] for s in(42,43,44)]
    mt,stt=st(tr);mv,sv=st(va);mp,sp=st(pp)
    print(f"{g:6} train {mt:.4f}+-{stt:.4f}  val {mv:.4f}+-{sv:.4f}  val_ppl {mp:.4f}+-{sp:.4f}")
gv=[res[f'bitnet_s{s}'][1]-res[f'stock_s{s}'][1] for s in(42,43,44)]
gt=[res[f'bitnet_s{s}'][0]-res[f'stock_s{s}'][0] for s in(42,43,44)]
mgv,sgv=st(gv);mgt,sgt=st(gt)
print(f"per-seed VAL gap {[round(x,4) for x in gv]} mean {mgv:.4f}+-{sgv:.4f}")
print(f"per-seed TRAIN gap {[round(x,4) for x in gt]} mean {mgt:.4f}+-{sgt:.4f}")
