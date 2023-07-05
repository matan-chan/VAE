# VAE with tensorflow

variational autoencoder implemnted with tesorflow compress images with shape of 256x256x3 into an array of 3000

## usage

```python
from model import VAE

vae = VAE(True)

z = vae.compress(images)  # compress the image to array of size 3000
# do stuff with z
z = vae.expand(z)  # back to image with shape 256x256x3

```

## training:

first run the `download_images()` function
then call the `train()` function.

```python
from main import VAEInterface

v = VAEInterface()
v.train()
```

## example:

<p align="left">
  <img width="800" src="https://github.com/matan-chan/VAE/blob/main/output_images/generated_plot_epoch-158000.png?raw=true">
</p>

## data:

[LAION5B][website]


[website]: https://laion.ai/blog/laion-5b/