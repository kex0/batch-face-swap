## Batch Face Swap extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui
 Automaticaly detects faces and replaces them.
 
![preview](https://user-images.githubusercontent.com/46696708/236370022-1e243f62-a59c-437a-a841-d9a3d37778aa.png)

## Installation
### Automatic:
1. In the WebUI go to `Extensions`.
2. Open `Available` tab and click `Load from:` button.
3. Find `Batch Face Swap` and click `Install`.
4. Apply and restart UI
### Manual:
1. Use `git clone https://github.com/kex0/batch-face-swap.git` from your SD web UI `/extensions` folder. 
2. Open `requirements_versions.txt` in the main SD web UI folder and add `mediapipe`.
3. Start or reload SD web UI.

## txt2img Guide
1. Expand the `Batch Face Swap` tab in the lower left corner.
    <details>
    <summary>Image</summary>
   <img src="https://user-images.githubusercontent.com/46696708/236360445-7391d68b-4973-4b43-aa75-f8a8782f6c4e.png">
    </details>
2. Click the checkbox to enable it.
    <details>
    <summary>Image</summary>
   <img src="https://user-images.githubusercontent.com/46696708/236361252-5a1d05c8-e216-4685-a7ef-80733e08a08a.png">
    </details>
3. Click `Generate`

## img2img Guide
1. Expand the `Batch Face Swap` tab in the lower left corner.
    <details>
    <summary>Image</summary>
   <img src="https://user-images.githubusercontent.com/46696708/236361645-84519cfe-d6a1-492f-adab-3baca037b6de.png">
    </details>
2. Click the checkbox to enable it.
    <details>
    <summary>Image</summary>
   <img src="https://user-images.githubusercontent.com/46696708/236361252-5a1d05c8-e216-4685-a7ef-80733e08a08a.png">
    </details>
3. You can process either 1 image at a time by uploading your image at the top of the page.
    <details>
    <summary>Image</summary>
   <img src="https://user-images.githubusercontent.com/46696708/236361988-78cfe787-d17a-46a1-bb41-4865a57dcdda.png">
    </details>
    Or you can give it path to a folder containing your images.
    <details>
    <summary>Image</summary>
   <img src="https://user-images.githubusercontent.com/46696708/236362301-53ce2315-9fa1-46e3-9698-d9164e1354be.png">
    </details>
4. Click `Generate`

Override options only affect face generation so for example in `txt2img` you can generate the initial image with one prompt and face swap with another. Or generate the initial image with one model and faceswap with another.
<details>
<summary>Example</summary>

Left 'young woman in red dress' using `chilloutMix`
Right 'Emma Watson in red dress' using `realisticVision`
<img src="https://user-images.githubusercontent.com/46696708/236363435-07e1cc38-062b-4696-9ce3-11239812f898.png">
</details>

![chrome_XSjamNtABV](https://user-images.githubusercontent.com/46696708/236360114-bd902f03-73cb-4836-a647-27f5371e3197.png)

## Example
![example](https://user-images.githubusercontent.com/46696708/211818536-7d3bd06e-f6b1-40e9-854e-9cb44be3b2f8.png)

Prompt:
```ShellSession
detailed closeup photo of Emma Watson, 35mm, dslr
Negative prompt: (painting:1.3), (concept art:1.2), artstation, sketch, illustration, drawing, blender, octane, 3d, render, blur, smooth, low-res, grain, cartoon, watermark, text, out of focus
```
