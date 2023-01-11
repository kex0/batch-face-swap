## Batch Face Swap for https://github.com/AUTOMATIC1111/stable-diffusion-webui
 Automaticaly detects faces and replaces them.

## Requirements
MediaPipe Python package
To install it, just open `requirements_versions.txt` located in your stable-diffusion-webui folder and add `mediapipe`

## Installation
**[Download the zipped script Here](https://github.com/kex0/batch-face-swap/archive/refs/heads/main.zip)**
and copy the file `face_swap.py` into your scripts folder.

## Guide
1. Open `img2img` tab. (you don't have to go to the `Inpaint` tab, if you do, you have to set "Mask source" to "Upload mask" or generation won't work)
2. Select `Batch Face Swap` script.
3. Paste a path of the folder containing your images in the `Images directory` textbox.
4. (Optional) It may sometimes fail to find a face if the face is very small in comparison to the size of the image.
So, you can tell it to split the image and look at the smaller portions of the image by using the `How many times to divide image` slider.
(don't worry it will stitch the image back together)

If you want to adjust `Denoising strength` or `Mask blur` you have to disable the override checkboxes.

You can generate and save the masks without even engaging the stable diffusion image generation by checking the `Save masks to disk` checkbox and pressing 
`Generate masks` button at the very bottom.

Tip:
You can check how many faces do your current settings find before you start generating with the stable diffusion 
by just clicking the `Generate masks` button at the very bottom.