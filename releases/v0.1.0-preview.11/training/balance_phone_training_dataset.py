"""Oversample existing positive training images in a generated YOLO dataset."""
import argparse, hashlib, json, shutil
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--base',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--positive-copies',type=int,default=1)
args=parser.parse_args()
if args.positive_copies<1: raise SystemExit('--positive-copies must be positive')
if args.output.exists(): raise SystemExit(f'output exists: {args.output}')
shutil.copytree(args.base,args.output)
split=json.loads((args.output/'split.json').read_text(encoding='utf-8'))
added=[]
for record in list(split['records']):
    if record.get('split')!='train' or not record.get('objects'):
        continue
    image=args.output/'images/train'/record['file_name']
    label=args.output/'labels/train'/(Path(record['file_name']).stem+'.txt')
    for copy_index in range(1,args.positive_copies+1):
        name=f'positive_repeat_{copy_index:02d}_{Path(record["file_name"]).name}'
        target=args.output/'images/train'/name
        shutil.copy2(image,target)
        shutil.copy2(label,args.output/'labels/train'/(Path(name).stem+'.txt'))
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        added.append({**record,'file_name':name,'source_file_name':record['file_name'],'source_sha256':digest,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source_run':'POSITIVE_REPEAT_'+str(copy_index)})
split['records'].extend(added)
split['counts']['train']['images']+=len(added)
split['counts']['train']['positive']+=len(added)
split['positive_repeat_audit']={'copies_per_original':args.positive_copies,'added_images':len(added),'source':'Existing historical positive train images only; owner acceptance images excluded.'}
(args.output/'split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
(args.output/'dataset.yaml').write_text(f'path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: cell_phone\n',encoding='utf-8')
print(json.dumps({'train_images':split['counts']['train']['images'],'positive':split['counts']['train']['positive'],'negative':split['counts']['train']['negative'],'added':len(added)}))
