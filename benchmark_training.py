"""Short disposable-model CUDA training benchmark; never writes model weights."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from xiangqi_cnn import PieceNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    model = PieceNet().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batch = 128
    x = torch.rand(batch,3,48,48,device='cuda')
    y = torch.randint(0,15,(batch,),device='cuda')
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x),y)
        loss.backward()
        optimizer.step()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    started=time.perf_counter()
    for _ in range(40):
        step()
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    report={'gpu':torch.cuda.get_device_name(), 'batch_size':batch,'steps':40,
            'seconds':seconds,'samples_per_second':batch*40/seconds,
            'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
            'note':'FP32 synthetic resident-tensor training; excludes data loading, validation and augmentation. Disposable random model; no checkpoint overwritten.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
