# Wand

Here you can find the detector object that finds a wand given a video stream, and the SpellNet trained small newural network that classifies the sigils

![](https://i.imgur.com/uQiWFLs.png)

## Detector

Using OpenCV, the WandDetector object creates an eternal loop until locates  the top five most bright circular objects after a blur and black and white.
Once these points are located, their movement is tracked, and the most complex sequence is stored in a sepparate object, called a sigil and returned to the controller.
See the code for further details.


## SpellNet

Essentially, a very single single-module Conv2D network and a couple of Dense layes. See the [Jupyter book](./spell_net/spell_net.ipynb) for details.
