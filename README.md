# Image Visualization Tools

![badge](https://img.shields.io/github/package-json/v/AndreGraca98/vis_tools?filename=vis_tools%2Fversion.json&label=vis-tools&logo=python&logoColor=yellow)

vis_tools is a library created to unify the different ways to read, write and display images ([opencv](https://github.com/opencv/opencv), [matplotlib](https://github.com/matplotlib/matplotlib), [PIL](https://github.com/python-pillow/Pillow), ...)

## Features

- Read, write and show single images, as well as multiple images
- Can use torch.Tensor, numpy.ndarray or a list of either of them to use the functions for multiple images
- Extract images from a video/directory with videos and save them to a folder
- Create a video from a directory of images

## Instalation

```bash
envname=vis
conda create -n $envname python=3.7 -y
conda activate $envname
conda install pytorch==1.12.0 -c pytorch -y
pip install git+https://github.com/AndreGraca98/vis_tools.git

```

## How to use

### 1. matplotlib

```python
from vis_tools import generate_gradient_2d, PLT

img = generate_gradient_2d()

# Write
PLT.write(img, 'img_PLT.png') # Note that matplotlib requires floats to be in range [0., 1.] 
```

```python
# Read
img_PLT = PLT.read('im_PLT.png')
```

```python
# Show
img_PLT = PLT.show(img_PLT, figsize=5)
```

### 2. opencv

```python
from vis_tools import generate_gradient_2d, CV2

img = generate_gradient_2d()

# Write
CV2.write((img*255).astype('uint8'), 'img_CV2.png')
```

```python
# Read
img_CV2 = CV2.read('im_CV2.png')
```

```python
# Show
img_CV2 = CV2.show(img_CV2, (224, 224)) 
```

### 3. PIL

```python
from vis_tools import generate_gradient_2d

img = generate_gradient_2d()

# Write
```

```python
# Read
```

```python
# Show
```

## TODO

  1. [x] Implement matplotlib functions
  1. [x] Implement opencv functions
  1. [ ] Implement PIL functions
  1. [ ] Add more examples to repo
  1. [ ] Add tests
