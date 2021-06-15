import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    print(images.shape)
    return padder.pad(images)[0]
        

def viz(img, flo, i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    output_flo_file = open('./prediction_imgs/' + str(i).zfill(4) + '.flo','wb')
    np.array([ 80, 73, 69, 72 ], np.uint8).tofile(output_flo_file)
    np.array([ 1280, 720 ], np.int32).tofile(output_flo_file)
    flo.tofile(output_flo_file)
    output_flo_file.close()    

    # map flow to rgb image
    #flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)

    #img_dir = './prediction_imgs/' + str(i) +'.png'
    #saved_img_flo = img_flo
    #cv2.imwrite(img_dir, saved_img_flo)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = load_image_list(images)
        for i in range(images.shape[0]-1):
            image1 = images[i,None]
            image2 = images[i+1,None]

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
