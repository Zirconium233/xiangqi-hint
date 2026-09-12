"""CUDA fine-tuning with grouped validation, gold regression gate and safe deployment."""
import argparse,json,random,time,shutil,os
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2,numpy as np,torch
from xiangqi_cnn import PieceNet,CLASSES,MODEL_PATH
from prepare_dataset import read_image

ROOT=Path(__file__).resolve().parent

def load(path,device):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    images=[]
    for r in rows:
        im=read_image(Path(r['file']))
        if im is None:raise RuntimeError('Unreadable training sample')
        images.append(cv2.cvtColor(cv2.resize(im,(48,48)),cv2.COLOR_BGR2RGB).transpose(2,0,1))
    return torch.tensor(np.ascontiguousarray(np.stack(images)),device=device,dtype=torch.float32)/255,torch.tensor([CLASSES.index(r['label']) for r in rows],device=device),rows

def evaluate(model,x,y,rows):
    model.eval()
    with torch.no_grad():
        logits=torch.cat([model(b) for b in x.split(128)])
        confidence,pred=logits.softmax(1).max(1)
    per={c:float((pred[y==i]==i).float().mean()) for i,c in enumerate(CLASSES) if (y==i).any()}
    errors=[{'index':i,'expected':CLASSES[int(y[i])],'predicted':CLASSES[int(pred[i])],'confidence':float(confidence[i]),'case':rows[i].get('case')} for i in range(len(rows)) if pred[i]!=y[i]]
    return {'accuracy':float((pred==y).float().mean()),'macro_recall':sum(per.values())/len(per),'per_class':per,
            'min_confidence':float(confidence.min()),'below_090':int((confidence<.90).sum()),'errors':errors,'size':len(rows)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);p.add_argument('--deploy',action='store_true');a=p.parse_args()
    random.seed(41);np.random.seed(41);torch.manual_seed(41);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for bounded training time')
    device='cuda'; data=ROOT/'datasets/prepared/batch1_templates/training_v2'
    trainx,trainy,trainrows=load(data/'train.jsonl',device)
    valx,valy,valrows=load(data/'val.jsonl',device)
    goldx,goldy,goldrows=load(ROOT/'datasets/prepared/gold_eval/gold.jsonl',device)
    if len(valrows)<100:raise RuntimeError('Validation set incomplete; finish annotation first')
    assert not ({r['id'] for r in trainrows}&{r['id'] for r in valrows})
    if len(set(trainy.tolist()))!=15:raise RuntimeError('Training classes incomplete')
    out=ROOT/'datasets/runs'/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    model=PieceNet().to(device);model.load_state_dict(torch.load(MODEL_PATH,map_location=device,weights_only=True))
    baseline_val=evaluate(model,valx,valy,valrows);baseline_gold=evaluate(model,goldx,goldy,goldrows)
    # Separate bounded-offset stress check; not used for epoch selection.
    padded=torch.nn.functional.pad(goldx,(3,3,3,3),mode='replicate')
    stressx=torch.cat([padded[:,:,3+dy:51+dy,3+dx:51+dx] for dx,dy in [(3,0),(-3,0),(0,3),(0,-3)]]).contiguous()
    stressy=goldy.repeat(4);stressrows=goldrows*4
    baseline_stress=evaluate(model,stressx,stressy,stressrows)
    print('Baseline',json.dumps({'val':baseline_val['accuracy'],'gold':baseline_gold['accuracy']}),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs)
    counts=torch.bincount(trainy,minlength=15).float();weights=1/counts[trainy]
    best=-1;history=[];started=time.monotonic()
    for epoch in range(a.epochs):
        model.train();indices=torch.multinomial(weights,len(trainy),replacement=True)
        loss_total=0
        for idx in indices.split(128):
            x=trainx[idx];y=trainy[idx];n=len(idx)
            theta=torch.zeros(n,2,3,device=device)
            scale=torch.empty(n,device=device).uniform_(.94,1.06)
            theta[:,0,0]=scale;theta[:,1,1]=scale;theta[:,:,2]=torch.empty(n,2,device=device).uniform_(-.10,.10)
            grid=torch.nn.functional.affine_grid(theta,x.shape,align_corners=False)
            x=torch.nn.functional.grid_sample(x,grid,padding_mode='border',align_corners=False)
            x=(x*torch.empty(n,1,1,1,device=device).uniform_(.90,1.10)).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x),y)
            loss.backward();optimizer.step();loss_total+=float(loss)
        scheduler.step();metrics=evaluate(model,valx,valy,valrows)
        score=metrics['macro_recall']+.001*metrics['accuracy']
        if score>best:
            best=score;torch.save(model.state_dict(),out/'candidate.pt')
        history.append({'epoch':epoch+1,'loss_sum':loss_total,'val_accuracy':metrics['accuracy'],'val_macro':metrics['macro_recall']})
        print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(torch.load(out/'candidate.pt',map_location=device,weights_only=True))
    final_val=evaluate(model,valx,valy,valrows);final_gold=evaluate(model,goldx,goldy,goldrows)
    final_stress=evaluate(model,stressx,stressy,stressrows)
    oldcases=Counter(e['case'] for e in baseline_gold['errors']);newcases=Counter(e['case'] for e in final_gold['errors'])
    gate=(final_val['macro_recall']>=baseline_val['macro_recall'] and len(final_gold['errors'])<=len(baseline_gold['errors'])
          and all(newcases[k]<=oldcases[k] for k in newcases)
          and final_gold['below_090']<=baseline_gold['below_090']
          and len(final_stress['errors'])<=len(baseline_stress['errors'])
          and (final_val['macro_recall']>baseline_val['macro_recall'] or len(final_gold['errors'])<len(baseline_gold['errors'])))
    report={'baseline_val':baseline_val,'baseline_gold':baseline_gold,'final_val':final_val,'final_gold':final_gold,
            'seconds':time.monotonic()-started,'history':history,'gate_passed':gate,'deployed':False,'train_samples':len(trainy)}
    report['validation_missing_classes']=[c for c in CLASSES if c not in final_val['per_class']]
    report['baseline_stress']=baseline_stress;report['final_stress']=final_stress
    report['data_manifest']=str(data)
    if a.deploy and gate:
        shutil.copy2(MODEL_PATH,out/'previous.pt')
        temporary=Path(MODEL_PATH).with_suffix('.pending.pt')
        shutil.copy2(out/'candidate.pt',temporary);os.replace(temporary,MODEL_PATH)
        report['deployed']=True
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'run':str(out),'seconds':report['seconds'],'deployed':report['deployed'],'gold_old_errors':len(baseline_gold['errors']),'gold_new_errors':len(final_gold['errors']),'val_accuracy':final_val['accuracy']}),flush=True)

if __name__=='__main__':main()
