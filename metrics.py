import argparse
import json
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
import cv2
import numpy as np
from os import listdir
from statistics import mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image quality.')
    parser.add_argument('--datasets_dir', type=str)
    parser.add_argument('--default_root_dir', type=str) 
    parser.add_argument('--original_datasets', type=str) 
    parser.add_argument('--predict_datasets', type=str) 

    args = parser.parse_args()
    original_dir = f'{args.datasets_dir}/{args.original_datasets}'
    sr_dir = f'{args.default_root_dir}/{args.predict_datasets}'

    original_imgs = sorted(listdir(original_dir))
    original_imgs = [x.split('.')[0] for x in original_imgs] # remove extensions

    ssim_scores = []
    psnr_scores = []
    for i in range(len(original_imgs)): # Both should have the same size
        before = cv2.imread(f'{original_dir}/{original_imgs[i]}')
        after = cv2.imread(f'{sr_dir}/{original_imgs[i]+".png"}')

        # Convert images to grayscale
        # before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        # after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        (ssim_score, ssim_diff) = calculate_ssim(before, after, channel_axis=2, full=True)
        ssim_scores.append(ssim_score)
        psnr_score = calculate_psnr(before, after)
        psnr_scores.append(psnr_score)

    # Data to be written
    psnr = {
        'scores': psnr_scores,
        'mean': mean(psnr_scores),
        'max': psnr_scores.index(max(psnr_scores))
    }
    ssim = {
        'scores': ssim_scores,
        'mean': mean(ssim_scores),
        'max': ssim_scores.index(max(ssim_scores))
    }
    data = {
        'psnr' : psnr,
        'ssim': ssim
    }

    print(f'--- PSNR ---')
    print(f'Mean: {psnr["mean"]}')
    print(f'Max: {psnr["scores"][psnr["max"]]}, {psnr["max"]})th image (0-indexed)')
    print(f'--- SSIM ---')
    print(f'Mean: {ssim["mean"]}')
    print(f'Max: {ssim["scores"][ssim["max"]]}, {ssim["max"]})th image (0-indexed)')

    # Serializing json
    json_object = json.dumps(data, indent=4)
    
    # Writing to sample.json
    with open(f'{args.default_root_dir}/metrics.json', "w") as outfile:
        outfile.write(json_object)

