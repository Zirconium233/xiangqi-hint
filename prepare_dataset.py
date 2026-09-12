"""Offline, non-destructive capture audit and annotation staging."""
import argparse
import json
import hashlib
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    path.write_bytes(data.tobytes())


def overview(source, output, indices=None):
    files = sorted(source.glob('*.png'))
    indices = np.linspace(0, len(files)-1, min(24, len(files))).astype(int) if indices is None else indices
    sheet = np.zeros((int(np.ceil(len(indices)/4))*220, 4*330, 3), np.uint8)
    for slot, index in enumerate(indices):
        img = read_image(files[index])
        if img is None:
            continue
        thumb = cv2.resize(img, (330, 200))
        y, x = (slot//4)*220, (slot%4)*330
        sheet[y:y+200, x:x+330] = thumb
        cv2.putText(sheet, files[index].stem, (x+5, y+216), 0, .45, (255,255,255), 1)
    save_image(output/'overview.png', sheet)


def build(source, output):
    from board_geometry import detect_grid, _grid_quality
    from xiangqi_cnn import PieceClassifierCNN, CLASSES, PIECE_LIMITS
    output.mkdir(parents=True, exist_ok=False)
    overview(source, output)
    classifier = PieceClassifierCNN()
    metadata = {r['file']: r for r in map(json.loads, (source/'frames.jsonl').read_text(encoding='utf-8').splitlines())}
    files = sorted(source.glob('*.png'))
    counts, classes = Counter(), Counter()
    grid = shape = None
    known_crops, known_boards = set(), set()
    examples = {}
    started = time.monotonic()
    with (output/'frames.jsonl').open('w', encoding='utf-8') as frames, (output/'crops.jsonl').open('w', encoding='utf-8') as crops:
        for index, path in enumerate(files):
            im = read_image(path)
            record = {'source': str(path.resolve()), 'session': source.name, 'metadata': metadata.get(path.name), 'label_verified': False}
            status = 'unreadable'
            if im is not None:
                if shape != im.shape:
                    grid, shape = detect_grid(im), im.shape
                quality = _grid_quality(im, *grid) if grid else -1
                # Geometry is only reused for fixed-size captures with line evidence.
                # Low scores are quarantined, never repaired into a training board.
                record['grid_score'] = quality
                if grid and quality >= 18:
                    cols, rows = grid
                    record['cols'], record['rows'] = cols, rows
                    radius = int(min(np.median(np.diff(cols)), np.median(np.diff(rows)))*.40)
                    patches = [im[int(y)-radius:int(y)+radius, int(x)-radius:int(x)+radius] for y in rows for x in cols]
                    predictions = classifier._classify_patches_full(patches)
                    labels = [p[0] or '_' for p in predictions]
                    conf = [p[1] for p in predictions]
                    pieces = Counter(labels)
                    valid = pieces['K'] == pieces['k'] == 1 and all(pieces[k] <= n for k,n in PIECE_LIMITS.items())
                    confident = min(conf) >= .90
                    status = 'static_candidate_review' if valid and confident else 'animation_or_uncertain_review'
                    record.update(pseudo_board=labels, min_confidence=min(conf), piece_counts=dict(pieces))
                    board_key = ''.join(labels)
                    record['repeated_pseudo_position'] = board_key in known_boards
                    if status == 'static_candidate_review':
                        known_boards.add(board_key)
                        for cell, (patch, prediction) in enumerate(zip(patches, predictions)):
                            small = cv2.resize(patch, (48,48))
                            key = hashlib.sha256((small//8).tobytes()).hexdigest()
                            if key in known_crops:
                                continue
                            known_crops.add(key)
                            name = 'unlabelled_crops/'+key+'.png'
                            save_image(output/name, patch)
                            p, confidence, probs = prediction
                            top = np.argsort(probs)[-3:][::-1]
                            crops.write(json.dumps({'file':name, 'source_frame':path.name, 'session':source.name,
                                'row':cell//9, 'col':cell%9, 'radius':radius,
                                'pseudo_label':p or '_', 'confidence':confidence,
                                'top3':[(CLASSES[int(i)],float(probs[i])) for i in top],
                                'label':None, 'verified':False, 'split':'unassigned'}, ensure_ascii=False)+'\n')
                            classes[p or '_'] += 1
                else:
                    status = 'no_grid_or_obscured_review'
            record['status'] = status
            counts[status] += 1
            examples.setdefault(status, [])
            if len(examples[status]) < 12:
                examples[status].append(path)
            frames.write(json.dumps(record, ensure_ascii=False)+'\n')
            if index % 100 == 0:
                print(f'{index}/{len(files)} {dict(counts)} crops={len(known_crops)}', flush=True)
    for status, paths in examples.items():
        sheet = np.zeros((int(np.ceil(len(paths)/4))*220,1320,3), np.uint8)
        for i,p in enumerate(paths):
            im = read_image(p)
            if im is not None:
                y,x=(i//4)*220,(i%4)*330
                sheet[y:y+200,x:x+330]=cv2.resize(im,(330,200))
                cv2.putText(sheet,p.stem,(x+5,y+216),0,.45,(255,255,255),1)
        save_image(output/(status+'.png'),sheet)
    report = {'frames':len(files),'status_counts':dict(counts),'unique_pseudo_positions':len(known_boards),
        'unique_crops':len(known_crops),'pseudo_class_counts':dict(classes),'seconds':time.monotonic()-started,
        'training_ready':False, 'reason':'Unverified pseudo-labels. Review labels and quarantine animations; split by game before training.',
        'geometry_policy':'First detected grid per image size, line-score gate >=18; manual audit required.',
        'source_bytes':sum(p.stat().st_size for p in files)}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--indices', help='Comma-separated indices for review contact sheet')
    args = parser.parse_args()
    if args.build:
        build(args.source, args.output)
    else:
        overview(args.source, args.output, [int(x) for x in args.indices.split(',')] if args.indices else None)
