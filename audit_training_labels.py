"""Contact sheet of weakest accepted labels, grouped by class."""
import json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
from prepare_dataset import read_image,save_image

def main():
    root=Path(__file__).resolve().parent/'datasets/prepared/batch1_templates/training_v2'
    grouped=defaultdict(list)
    for name in ['train','val']:
        for r in map(json.loads,(root/(name+'.jsonl')).read_text().splitlines()):grouped[r['label']].append(r)
    sheet=np.full((len(grouped)*74,840,3),245,np.uint8)
    for row,(label,items) in enumerate(sorted(grouped.items())):
        cv2.putText(sheet,label,(0,row*74+40),0,.6,(0,0,0),2)
        for col,r in enumerate(sorted(items,key=lambda r:r['score'])[:10]):
            x,y=40+col*80,row*74
            sheet[y:y+52,x:x+52]=cv2.resize(read_image(Path(r['file'])),(52,52))
            cv2.putText(sheet,str(round(r['score'],3)),(x,y+68),0,.35,(0,0,0),1)
    save_image(root/'weakest_labels.png',sheet)

if __name__=='__main__':main()
