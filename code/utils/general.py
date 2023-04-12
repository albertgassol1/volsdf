import datetime
import os
from glob import glob
from typing import Any, Dict, List

import torch


def get_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def mkdir_ifnotexists(directory: str) -> None:
    if not os.path.exists(directory):
        os.mkdir(directory)


def get_class(kls: str) -> object:
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def glob_imgs(path: str) -> List[Any]:
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def split_input(model_input: Dict[str, Any], total_pixels: int, n_pixels: int = 10000) -> List[Any]:
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split


def merge_output(res: List[Any], total_pixels: int, batch_size: int) -> Dict[str, Any]:
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs


def concat_home_dir(path: str) -> str:
    return os.path.join(os.environ['HOME'], 'data', path)
