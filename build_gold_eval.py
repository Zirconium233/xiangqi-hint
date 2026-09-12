"""Independent, manually transcribed historical screenshots. Never train on these."""
import json
from pathlib import Path
import cv2,numpy as np
from prepare_dataset import read_image,save_image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'datasets/prepared/gold_eval'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[
        ('midgame',Path('C:/Temp/Temp/codex-clipboard-1bdfe9eb-167d-4488-aa0f-901e5946abac.png'),
         (2048,836), (1260,1704,166,666),
         ['r___kab__','____a___r','_cn_bc___','p_p_p___p','______p__','__P_P____','P___N___P','_C__n_N_B','___R_____','__BAKA_R_']),
        ('endgame',Path('C:/Temp/Temp/codex-clipboard-f3d7c559-63be-4d87-97dc-6e600342ad64.png'),
         (1696,1023), (523,1171,143,870),
         ['____k____','___PaP___','_________','_________','_________','_________','_________','_____K___','_________','_________'])]
    rows=[]
    for name,path,(dw,dh),(x0,x1,y0,y1),board in cases:
        assert len(board)==10 and all(len(r)==9 for r in board)
        im=read_image(path);h,w=im.shape[:2]
        cols=np.linspace(x0*w/dw,x1*w/dw,9);ys=np.linspace(y0*h/dh,y1*h/dh,10)
        radius=int(min(np.diff(cols).mean(),np.diff(ys).mean())*.4)
        sheet=np.full((10*66,9*66,3),245,np.uint8)
        for r,y in enumerate(ys):
            for c,x in enumerate(cols):
                patch=im[int(y)-radius:int(y)+radius,int(x)-radius:int(x)+radius]
                target=OUT/f'{name}_{r}_{c}.png';save_image(target,patch)
                rows.append({'file':str(target),'label':board[r][c],'case':name,'row':r,'col':c,'method':'manual_visual_gold'})
                sheet[r*66:r*66+48,c*66:c*66+48]=cv2.resize(patch,(48,48))
                cv2.putText(sheet,board[r][c],(c*66+5,r*66+62),0,.4,(0,0,0),1)
        save_image(OUT/(name+'_review.png'),sheet)
    (OUT/'gold.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

if __name__=='__main__':main()
