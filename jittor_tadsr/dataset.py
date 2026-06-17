from pathlib import Path
import random, shlex
import numpy as np
from PIL import Image

def read_dataset_list(path):
    items=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        p=shlex.split(line)
        items.append({"hr":p[0], "lr":p[1], "prompt":p[2] if len(p)>2 else "a low-quality real-world image"})
    return items

def load_image(path, image_size):
    img=Image.open(path).convert("RGB").resize((image_size,image_size), Image.Resampling.BICUBIC)
    return (np.asarray(img).astype("float32")/255.0).transpose(2,0,1)

def sample_batch(items,batch_size,image_size):
    b=random.choices(items,k=batch_size)
    return np.stack([load_image(x["lr"],image_size) for x in b]), np.stack([load_image(x["hr"],image_size) for x in b]), [x["prompt"] for x in b], b

def iter_items(items,image_size):
    for item in items:
        yield load_image(item["lr"],image_size)[None], load_image(item["hr"],image_size)[None], item
