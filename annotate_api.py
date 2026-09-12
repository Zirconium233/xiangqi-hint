"""Resumable direct vision annotation. Credentials are read in memory only."""
import argparse,base64,json,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import cv2,numpy as np,requests,yaml
from prepare_dataset import read_image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'datasets/prepared/batch1_templates'
PROMPT='''Identify the visible Xiangqi symbol in each numbered tile. Return ONLY a JSON object {"labels":[...]} with exactly COUNT entries in tile order. Allowed labels: _ (empty grid, wood, or selection dot without a piece), R red chariot 車/车, N red horse 馬/马, B red elephant 相, A red advisor 仕, K red king 帥/帅, C red cannon 炮, P red soldier 兵; r black chariot 車/车, n black horse 馬/马, b black elephant 象, a black advisor 士, k black king 將/将, c black cannon 砲/炮, p black soldier 卒. Red lettering is dark red/brown; black lettering is dark blue/black. Use ? for obscured, partial moving pieces, effects, or ambiguous glyph/color. Do not infer a hidden piece. Ignore green move markers only if the underlying symbol remains clearly readable. Do not explain.'''

PROMPT = PROMPT.replace('{"labels":[...]} with exactly COUNT entries in tile order.', '{"labels":{"0":"label", "1":"label", ...}}. There are exactly COUNT tiles, numbered 0 through COUNT_MINUS_ONE. Read the ID BELOW each image carefully; include every ID, including empty tiles.')

def config():
    base=Path.home()/'.dsh'
    provider=yaml.safe_load((base/'settings.yaml').read_text(encoding='utf-8'))['llm-pi-ai']['providers']['qwen-vllm']
    credentials=yaml.safe_load((base/'.credentials.yaml').read_text(encoding='utf-8'))
    key=os.getenv(provider['apiKeyEnv']) or credentials['refs'][provider['apiKeyEnv']]
    return provider['baseURL'].rstrip('/')+'/chat/completions',key

def call(rows,url,key):
    sheet=np.full((int(np.ceil(len(rows)/4))*144,576,3),255,np.uint8)
    for i,r in enumerate(rows):
        im=read_image(DATA/r['file'])
        y,x=(i//4)*144,(i%4)*144
        sheet[y:y+110,x:x+110]=cv2.resize(im,(110,110))
        cv2.putText(sheet,'ID '+str(i),(x+3,y+135),0,.6,(0,0,0),2)
    ok,buf=cv2.imencode('.png',sheet)
    body={'model':'Qwen3.8-Flash-Next','temperature':0,'max_tokens':2048,
          'messages':[{'role':'user','content':[{'type':'text','text':PROMPT.replace('COUNT_MINUS_ONE',str(len(rows)-1)).replace('COUNT',str(len(rows)))},
          {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf).decode()}}]}]}
    for attempt in range(3):
        try:
            response=requests.post(url,headers={'Authorization':'Bearer '+key},json=body,timeout=(15,120))
            if response.status_code!=200:
                raise RuntimeError(f'API HTTP {response.status_code}')
            value=response.json()['choices'][0]['message']['content']
            value=value[value.find('{'):value.rfind('}')+1]
            mapping=json.loads(value)['labels']
            if not isinstance(mapping,dict) or set(mapping)!=set(map(str,range(len(rows)))):
                raise ValueError('Invalid tile IDs')
            labels=[mapping[str(i)] for i in range(len(rows))]
            if len(labels)!=len(rows) or any(x not in list('_RNBAKCPrnbakcp?') for x in labels):
                raise ValueError('Invalid label list')
            return [{'id':r['id'],'label':None if label=='?' else label,'model':'Qwen3.8-Flash-Next'} for r,label in zip(rows,labels)]
        except Exception as exc:
            if attempt==2:
                # Never include server body or request headers in diagnostics.
                raise RuntimeError(f'Annotation failed: {type(exc).__name__}') from None
            time.sleep(1+attempt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();url,key=config()
    target=DATA/'vision_labels_v2.jsonl'
    done={json.loads(x)['id'] for x in target.read_text().splitlines()} if target.exists() else set()
    rows=[json.loads(x) for x in (DATA/'labels.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['id'] not in done]
    # Grossly obscured/low-similarity crops remain unknown without API expense.
    rows=[r for r in rows if r['label'] is not None or (r['score']>=.80 and r['margin']>=.015)]
    if a.limit:rows=rows[:a.limit]
    batches=[rows[i:i+12] for i in range(0,len(rows),12)]
    with target.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(call,b,url,key) for b in batches]
        for i,future in enumerate(as_completed(futures)):
            for r in future.result():f.write(json.dumps(r)+'\n')
            f.flush();print(f'Batches {i+1}/{len(batches)}',flush=True)

if __name__=='__main__':main()
