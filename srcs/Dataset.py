import os, re, cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage

class Dataset:
    """
    Args:
        path      : Path to FGNET dataset directory e.g. './dataset/FGNET'
        transform : Optional torchvision transforms to apply to images
    """
    def __init__(self, path: str, transform=None):
        self.transform      = transform
        self.lbls_path      = os.path.join(path, 'Data_files')
        self.images_path    = os.path.join(path, 'images')
        self.pts_path       = os.path.join(path, 'points')
        self.df             = pd.DataFrame(self._load_())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        label   = torch.tensor([row['age']], dtype=torch.float32)
        image   = Image.open(row['img']).convert('RGB')

        pts = row['pts']
        if pts:
            xmin, ymin, xmax, ymax = self._get_bbox_from_pts_(pts)
            image = image.crop((xmin, ymin, xmax, ymax)) # Crop the image to leave only face features based on the pts file.

        if self.transform:
            image = self.transform(image)
        return image, label

    def _parse_lbls_(self, path):
        pattern     = '([0-9]+)a([0-9]+)([AaBb]?)$'
        name        = os.path.basename(path).split('.')[0]
        match       = re.match(pattern, name, re.IGNORECASE)

        if not match:
            return None
        return {
                'subject'   : match.group(1),
                'age'       : match.group(2),
                'id'        : match.group(3) if match.group(3) else 'a'
            }

    def _load_pts_(self, path):
        points      = []
        pos         = False

        if path is None or not os.path.exists(path):
            return None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == '{':
                    pos = True
                    continue
                if line == '}':
                    break
                if pos and line:
                    x, y = map(np.float64, line.split())
                    points.append({'x': x, 'y': y})
        return points

    def _get_bbox_from_pts_(self, pts, margin=20):
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]

        xmin = int(min(xs)) - margin
        ymin = int(min(ys)) - margin
        xmax = int(max(xs)) + margin
        ymax = int(max(ys)) + margin

        return xmin, ymin, xmax, ymax
    def _load_(self):
        records = []

        for fname in sorted(os.listdir(self.images_path)):
            if not fname.endswith('.jpg'):
                continue
            name        = os.path.splitext(fname)[0]
            img_path    = os.path.join(self.images_path, fname)
            pts_path    = os.path.join(self.pts_path, name + '.pts')
            label       = self._parse_lbls_(fname)
            if label is None:
                continue
            records.append({
                'subject'   : label['subject'],
                'age'       : np.int32(label['age']),
                'id'        : label['id'],
                'img'       : img_path,
                'pts'       : self._load_pts_(pts_path)
                })
        return records