import csv, random, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageDraw
def set_seed(seed): random.seed(seed); np.random.seed(seed)
def ensure_dirs(*paths):
    for p in paths: Path(p).mkdir(parents=True,exist_ok=True)
def save_image(arr,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Image.fromarray((np.clip(arr,0,1).transpose(1,2,0)*255).astype("uint8")).save(path)
def save_grid(lr,sr,hr,path):
    imgs=[Image.fromarray((np.clip(x,0,1).transpose(1,2,0)*255).astype("uint8")).resize((128,128)) for x in [lr,sr,hr]]
    imgs.append(ImageChops.difference(imgs[1],imgs[2])); labels=["LR","SR","HR","abs error"]
    can=Image.new("RGB",(512,152),"white"); d=ImageDraw.Draw(can)
    for i,img in enumerate(imgs): can.paste(img,(128*i,24)); d.text((128*i+4,4),labels[i],fill=(0,0,0))
    Path(path).parent.mkdir(parents=True,exist_ok=True); can.save(path)
def append_csv(path,row):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)
def write_json(path,obj): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
