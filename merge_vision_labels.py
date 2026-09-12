"""Only matching vision/template evidence enters training; conflicts stay unknown."""
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'datasets/prepared/batch1_templates'

def main():
    rows=[json.loads(x) for x in (ROOT/'labels.jsonl').read_text().splitlines()]
    vision={r['id']:r['label'] for r in map(json.loads,(ROOT/'vision_labels_v2.jsonl').read_text().splitlines())}
    groups=defaultdict(set)
    for r in map(json.loads,(ROOT/'occurrences.jsonl').read_text().splitlines()):
        n=int(Path(r['frame']).stem)
        if n<=430 or 450<=n<=1030:groups[r['id']].add('train')
        elif 1070<=n<=1490:groups[r['id']].add('val')
    out=ROOT/'training_v2';out.mkdir(exist_ok=True)
    counts=defaultdict(Counter); rejected=[]
    streams={s:(out/(s+'.jsonl')).open('w',encoding='utf-8') for s in ['train','val']}
    try:
        for r in rows:
            label=r['label'] or r['suggestion']
            trusted=r['method']=='verified_opening'
            matched=vision.get(r['id'])==label
            valid=trusted or (matched and (r['label'] is not None or (r['score']>=.80 and r['margin']>=.015 and r['pixel_error']<=.08)))
            if not valid:
                rejected.append({'id':r['id'],'file':str(ROOT/r['file']),'vision':vision.get(r['id']),'suggestion':r['suggestion'],'label':None})
                continue
            split='train' if 'train' in groups[r['id']] else ('val' if 'val' in groups[r['id']] else None)
            if split:
                entry={**r,'label':label,'file':str(ROOT/r['file']),'split':split,'evidence':'manual_opening' if trusted else 'vision_template_agreement'}
                streams[split].write(json.dumps(entry)+'\n');counts[split][label]+=1
    finally:
        for f in streams.values():f.close()
    (out/'unlabelled.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rejected),encoding='utf-8')
    report={'counts':dict(counts),'unlabelled':len(rejected),'vision_annotations':len(vision)}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
