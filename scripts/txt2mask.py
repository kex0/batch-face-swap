import sys
import os
import modules.scripts as scripts

base_dir = scripts.basedir()
sys.path.append(base_dir)

from clipseg.models.clipseg import CLIPDensePredT
from PIL import ImageChops, Image, ImageOps
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import numpy
from modules.images import flatten
from modules.shared import opts

def txt2mask(image, detectionPrompt):
	image = Image.fromarray(image)
	width, height = image.size
	delimiter_string = "|"

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	brush_mask_mode = "add"

	legacy_weights = False
	smoothing = 20
	smoothing_kernel = None
	if smoothing > 0:
		smoothing_kernel = numpy.ones((smoothing,smoothing),numpy.float32)/(smoothing*smoothing)

	# Pad the mask by applying a dilation or erosion
	mask_padding = 0
	padding_dilation_kernel = None
	if (mask_padding != 0):
		padding_dilation_kernel = numpy.ones((abs(mask_padding), abs(mask_padding)), numpy.uint8)

	prompts = detectionPrompt.split(delimiter_string)
	prompt_parts = len(prompts)

	mask_precision = 100


	def download_file(filename, url):
		import requests
		with open(filename,'wb') as fout:
			response = requests.get(url, stream=True)
			response.raise_for_status()
			# Write response data to file
			for block in response.iter_content(4096):
				fout.write(block)

	def overlay_mask_part(img_a,img_b,mode):
		if (mode == "discard"): img_a = ImageChops.darker(img_a, img_b)
		else: img_a = ImageChops.lighter(img_a, img_b)
		return(img_a)

	def gray_to_pil(img):
		return (Image.fromarray(cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)))

	def process_mask_parts(masks, mode, final_img = None, mask_precision=100, mask_padding=0, padding_dilation_kernel=None, smoothing_kernel=None):
		for i, mask in enumerate(masks):
			filename = f"mask_{mode}_{i}.png"
			plt.imsave(filename,torch.sigmoid(mask[0]))

			# TODO: Figure out how to convert the plot above to numpy instead of re-loading image
			img = cv2.imread(filename)

			if padding_dilation_kernel is not None:
				if (mask_padding > 0): img = cv2.dilate(img,padding_dilation_kernel,iterations=1)
				else: img = cv2.erode(img,padding_dilation_kernel,iterations=1)
			if smoothing_kernel is not None: img = cv2.filter2D(img,-1,smoothing_kernel)

			gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			(thresh, bw_image) = cv2.threshold(gray_image, mask_precision, 255, cv2.THRESH_BINARY)

			if (mode == "discard"): bw_image = numpy.invert(bw_image)

			# overlay mask parts
			bw_image = gray_to_pil(bw_image)

			if (i > 0 or final_img is not None): bw_image = overlay_mask_part(bw_image,final_img,mode)

			final_img = bw_image
		return(final_img)
		
	def get_mask(image):
		# load model
		model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=not legacy_weights)
		model_dir = f"{base_dir}/clipseg/weights"
		os.makedirs(model_dir, exist_ok=True)

		d64_filename = "rd64-uni.pth" if legacy_weights else "rd64-uni-refined.pth"
		d64_file = f"{model_dir}/{d64_filename}"
		d16_file = f"{model_dir}/rd16-uni.pth"

		# Download model weights if we don't have them yet
		if not os.path.exists(d64_file):
			print("Downloading clipseg model weights...")
			download_file(d64_file,f"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files={d64_filename}")
			download_file(d16_file,"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth")

		# non-strict, because we only stored decoder weights (not CLIP weights)
		model.load_state_dict(torch.load(d64_file), strict=False);	
		model = model.eval().to(device=device)

		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.Resize((512, 512)),
		])
		flattened_input = flatten(image, opts.img2img_background_color)
		img = transform(flattened_input).unsqueeze(0)

		# predict
		with torch.no_grad():
			preds = model(img.repeat(prompt_parts,1,1,1).to(device=device), prompts)[0].cpu()

		final_img = None

		# process masking
		final_img = process_mask_parts(preds,"add",final_img, mask_precision, mask_padding, padding_dilation_kernel, smoothing_kernel)

		return final_img

	image_mask = get_mask(image).resize((width,height))

	return image_mask