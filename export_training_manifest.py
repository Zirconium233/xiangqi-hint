"""Batch-1 reviewed game ranges; provisional template-label train/val staging."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from prepare_dataset import read_image,save_image


def run(root):
    labels=[json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines()]
    memberships=defaultdict(set)
    for line in (root/'occurrences.jsonl').read_text().splitlines():
        r=json.loads(line)
        frame=int(Path(r['frame']).stem)
        if frame<=430 or 450<=frame<=1030:
            memberships[r['id']].add('train')
        elif 1070<=frame<=1490:
            memberships[r['id']].add('validation_provisional')
    counts=defaultdict(Counter)
    paths={s:(root/(s+'.jsonl')).open('w',encoding='utf-8') for s in ['train','validation_provisional','unlabelled']}
    byclass=defaultdict(list)
    try:
        for r in labels:
            if r['label'] is None:
                split='unlabelled'
            elif 'train' in memberships[r['id']]:
                split='train'
            elif 'validation_provisional' in memberships[r['id']]:
                split='validation_provisional'
            else:
                continue
            r['split']=split
            r['file']=str((root/r['file']).resolve())
            paths[split].write(json.dumps(r)+'\n')
            counts[split][r['label'] or '?']+=1
            if r['label'] is not None:
                byclass[r['label']].append(r)
    finally:
        for f in paths.values():
            f.close()
    sheet=np.full((len(byclass)*70,10*70+90,3),240,np.uint8)
    for row,(label,items) in enumerate(sorted(byclass.items())):
        cv2.putText(sheet,label,(5,row*70+30),0,.7,(0,0,0),2)
        for col,r in enumerate(sorted(items,key=lambda x:x['score'])[:10]):
            x,y=90+col*70,row*70
            im=read_image(Path(r['file']))
            sheet[y:y+50,x:x+50]=cv2.resize(im,(50,50))
            cv2.putText(sheet,f"{r['score']:.3f}",(x,y+64),0,.32,(0,0,0),1)
    save_image(root/'accepted_weakest_review.png',sheet)
    summary={k:dict(v) for k,v in counts.items()}
    (root/'split_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    a=p.parse_args()
    run(a.root)
