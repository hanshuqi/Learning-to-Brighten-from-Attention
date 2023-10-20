import os
import argparse
import torch
from math import exp
import torchvision.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from torch.autograd import Variable

from model.Dual_Stream import UNet_base
from data_loader import PairLoader

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda', help="using GPU")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("--testDir", type=str, required=True, default='', help="path to test set")
parser.add_argument("--resultDir", type=str, default='', help="path to result")
parser.add_argument("--ckptDir", type=str, default='', help="path to checkpoint")
opt = parser.parse_args()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    img1=img1.clamp(0,1).to(opt.device)
    img2=img2.clamp(0,1).to(opt.device)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    return _ssim(img1, img2, window, window_size, channel, size_average)


def test_(model, test_loader, result_dir):
	torch.cuda.empty_cache()
	psnr_list = 0.0
	ssim_list = 0.0
	os.makedirs(os.path.join(result_dir), exist_ok=True)

	model.eval()
	for idx, val_data in enumerate(test_loader):

		with torch.no_grad():
			haze, gt, image_name = val_data['low'], val_data['normal'], val_data['filename']
			haze = haze.to(opt.device)
			gt = gt.to(opt.device)
			_, _, pred = model(haze)

			pred = pred.clamp_(-1, 1) * 0.5 + 0.5
			gt = gt * 0.5 + 0.5

			# --- Calculate the average PSNR --- #
			mse_loss = F.mse_loss(pred, gt, reduction='none').mean((1, 2, 3))
			psnr_tmp = 10 * torch.log10(1 / mse_loss).mean().item()
			psnr_list += psnr_tmp

			# --- Calculate the average SSIM --- #
			ssim_tmp=ssim(pred, gt).item()
			ssim_list += ssim_tmp

			tmp = 'Image:{}\tPSNR={:.5f}\tSSIM={:.5f}\n'.format(
				image_name[0], psnr_tmp, ssim_tmp)
		print(tmp)

		path = result_dir + '/' + image_name[0].split('.')[0] + 'psnr{:.5f}_ssim{:.5f}.png'.format(psnr_tmp, ssim_tmp)
		utils.save_image(pred[0], path)

	avr_psnr = psnr_list / len(test_loader)
	avr_ssim = ssim_list / len(test_loader)
	tmp = 'Average Results: psnr{:.5f}_ssim{:.5f}\n'.format(avr_psnr, avr_ssim)
	with open('test_logging_1223.txt', "a") as file:
		file.write(tmp)
	print(tmp)


if __name__ == '__main__':

	result_dir = opt.resultDir
	test_set = opt.testDir
	ckpt_path = opt.ckptDir
	if opt.device == 'cuda':
		torch.cuda.current_device()
		torch.cuda.empty_cache()
		torch.cuda._initialized = True
		torch.cuda.set_device(0)

	ckp = torch.load(ckpt_path, map_location=opt.device)
	model = UNet_base()

	model.load_state_dict(ckp['model'])

	model = model.to(opt.device)

	test_data = PairLoader(test_set, 'test', 'test')
	test_loader = DataLoader(test_data, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)

	os.makedirs(result_dir, exist_ok=True)

	test_(model, test_loader, result_dir)