## Batch Face Swap for https://github.com/AUTOMATIC1111/stable-diffusion-webui
 Automaticaly detects faces and replaces them.

## Requirements
MediaPipe Python package
To install it, just open `requirements_versions.txt` located in your stable-diffusion-webui folder and add `mediapipe`

## Installation and Requirements
The face detection requires MediaPipe Python package
To install it, just open `requirements_versions.txt` located in your stable-diffusion-webui folder and add `mediapipe`

To install the script just download the zipped script **[Here](https://github.com/kex0/batch-face-swap/archive/refs/heads/main.zip)**
and copy the file `face_swap.py` into your scripts folder.

## Guide
1. Open `img2img` tab. (you don't have to go to the `Inpaint` tab, if you do, you have to set "Mask source" to "Upload mask" or generation won't work)
2. Select `Batch Face Swap` script.
3. Paste a path of the folder containing your images in the `Images directory` textbox.
4. (Optional) It may sometimes fail to find a face if the face is very small in comparison to the size of the image.
So, you can tell it to split the image and look at the smaller portions of the image by using the `How many times to divide image` slider.
(don't worry it will stitch the image back together)
5. Click `Generate`

If you want to adjust `Denoising strength` or `Mask blur` you have to disable the override checkboxes.

You can generate and save the masks without even engaging the stable diffusion image generation by checking the `Save masks to disk` checkbox and pressing 
`Generate masks` button at the very bottom.

Tip:
You can check how many faces do your current settings find before you start generating with the stable diffusion 
by just clicking the `Generate masks` button at the very bottom.

## Example
![example](https://user-images.githubusercontent.com/46696708/211818536-7d3bd06e-f6b1-40e9-854e-9cb44be3b2f8.png)

```ShellSession
detailed closeup photo of Emma Watson, 35mm, dslr
Negative prompt: (painting:1.3), (concept art:1.2), artstation, sketch, illustration, drawing, blender, octane, 3d, render, blur, smooth, low-res, grain, cartoon, watermark, text, out of focus
Steps: 50, Sampler: Euler a, CFG scale: 7, Seed: 4052732944, Size: 512x512, Model hash: a9263745, Batch size: 8, Batch pos: 1, Denoising strength: 0.5, Mask blur: 4
```