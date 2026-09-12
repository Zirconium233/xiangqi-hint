"""Conservative labels from an explicitly verified opening frame, not CNN truth."""
import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from prepare_dataset import read_image, save_image

OPENING = 'rnbakabnr'+'_'*9+'_c_____c_'+'p_p_p_p_p'+'_'*18+'P_P_P_P_P'+'_C_____C_'+'_'*9+'RNBAKABNR'


def patches(im, grid):
    cols, rows = grid
    radius = int(min(np.median(np.diff(cols)),np.median(np.diff(rows)))*.4)
    return [im[int(y)-radius:int(y)+radius,int(x)-radius:int(x)+radius] for y in rows for x in cols]


def run(source, audit, output):
    output.mkdir(parents=True,exist_ok=False)
    records=[json.loads(line) for line in (audit/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    first=records[0]
    grid=(first['cols'],first['rows'])
    reference=patches(read_image(Path(first['source'])),grid)
    templates=[cv2.resize(p,(48,48))[4:-4,4:-4] for p in reference]
    symbols=sorted(set(OPENING))
    by_class={c:[t for t,label in zip(templates,OPENING) if c==label] for c in symbols}
    counts=Counter()
    seen={}
    galleries={}
    with (output/'labels.jsonl').open('w',encoding='utf-8') as manifest, (output/'occurrences.jsonl').open('w',encoding='utf-8') as occurrences:
        for index,record in enumerate(records):
            if 'cols' not in record:
                continue
            im=read_image(Path(record['source']))
            for cell,patch in enumerate(patches(im,(record['cols'],record['rows']))):
                resized=cv2.resize(patch,(48,48))
                key=hashlib.sha256(resized.tobytes()).hexdigest()
                occurrences.write(json.dumps({'id':key,'frame':Path(record['source']).name,'row':cell//9,'col':cell%9,'session':source.name})+'\n')
                if key in seen:
                    continue
                scores=[]
                for c in symbols:
                    score=max(float(cv2.minMaxLoc(cv2.matchTemplate(resized,t,cv2.TM_CCOEFF_NORMED))[1]) for t in by_class[c])
                    scores.append((score,c))
                scores.sort(reverse=True)
                score,label=scores[0]
                margin=score-scores[1][0]
                # Colour-aware pixel error disambiguates red/black look-alikes.
                error=min(float(cv2.minMaxLoc(cv2.matchTemplate(resized,t,cv2.TM_SQDIFF_NORMED))[0]) for t in by_class[label])
                accepted=score>=.96 and margin>=.045 and error<=.025
                if index==0:
                    label=OPENING[cell]
                    accepted=True
                bucket=label if accepted else 'unlabelled'
                folder={'_':'empty'}.get(bucket,('red_'+bucket if bucket.isupper() else 'black_'+bucket)) if accepted else bucket
                name='crops/'+folder+'/'+key+'.png'
                save_image(output/name,patch)
                row={'id':key,'file':name,'label':label if accepted else None,'suggestion':label,
                     'method':'verified_opening' if index==0 else ('template_high_precision' if accepted else 'needs_review'),
                     'score':score,'margin':margin,'pixel_error':error,'source_frame':Path(record['source']).name,
                     'row':cell//9,'col':cell%9,'session':source.name,'split':'unassigned',
                     'reference_frame':Path(first['source']).name}
                manifest.write(json.dumps(row)+'\n')
                seen[key]=row
                counts[bucket]+=1
                galleries.setdefault(bucket,[])
                # Include the weakest accepted matches for inspection later.
                galleries[bucket].append(row)
            if index%200==0:
                print(index,dict(counts),flush=True)
    for bucket,items in galleries.items():
        items=sorted(items,key=lambda r:r['score'])[:100]
        sheet=np.full((int(np.ceil(len(items)/10))*75,750,3),245,np.uint8)
        for i,row in enumerate(items):
            y,x=(i//10)*75,(i%10)*75
            patch=read_image(output/row['file'])
            sheet[y:y+55,x:x+55]=cv2.resize(patch,(55,55))
            cv2.putText(sheet,f"{row['suggestion']} {row['score']:.2f}",(x,y+69),0,.30,(0,0,0),1)
        # Windows paths are case-insensitive: prefix colour explicitly.
        name='empty' if bucket=='_' else ('red_'+bucket if bucket.isupper() else 'black_'+bucket)
        save_image(output/('review_'+name+'.png'),sheet)
    report={'unique_crops':len(seen),'counts':dict(counts),'accepted':sum(v for k,v in counts.items() if k!='unlabelled'),
            'unlabelled':counts['unlabelled'],'reference':'000000.png visually verified standard red-bottom opening',
            'policy':'No CNN-generated ground truth. Strict template labels are provisional pending sampled audit.',
            'split_policy':'Not split yet. Partition by verified game boundaries; use occurrences to prevent duplicate leakage.'}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('audit',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.source,a.audit,a.output)
