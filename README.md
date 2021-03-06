# PiPotter

A raspberry pi camera wand reader, loosely based, yet inspired,  on [Rpotter](https://github.com/sean-obrien/rpotter/blob/master/rpotter.py)


## Architecture

![](https://i.imgur.com/m5g8WOs.jpg)

The short version: Using a PiNoir camera, once an IR-reflective wand is detected, a 3 seconds movement pattern is recorded. That pattern, a sigil from now on, is passed trough a pre-trained small neural network (SpellNet) and the identified spell is sent to a controller that runs an effect, that may be lights (using the PythonMilight interface) or sound (mplayer)
The detection does looks like this, roughly:

![](https://i.imgur.com/uQiWFLs.png)



## Installation

Firstly, clone the env:
```bash
git clone https://github.com/Esiravegna/pipotter.git
```

### Required hardware

Raspberry Pi 3. It may work in a 2 or Zero, however, I did not test it. Also,a pinoir camera with led illuminators (the latter may be any other source) are needed. In theory, any IR enabled, picamera compatible camera would do, however, I did not test it.
A sound output for the aforementioned RBpi, and a set of lights based on MiLight or LimitlessLED.

Remmeber to have [conda](https://conda.io/docs/index.html) installed on the training machine, or [berryconda](https://github.com/jjhelmus/berryconda) on the raspberry pi

### On the training machine
```bash
conda env create -f pipotter_training.yaml
```

### On the raspberry pi:

```bash
conda env create -f pipotter.yaml
```
The TensorFlow part will be tricky, as the [working images](http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/) are available for python 3.4, while berryconda provides 3.6. Once you download the wheel, proceed to:
```bash
wheel install /the/path/to/the/tensorflow.whl
``` 
and then:
```bash
pip install protobuf absl-py tensorflow-tensorboard
```

Et voila!

## Usage

Once your prefered effects are set, you can run the server such as:

```bash
 ./run.sh --video-source=picamera
```

This would run the server with the provided config.json. Try

```bash
./run.sh --help
```

for a list of commands.

## Where to go from here

Please refer to each directory for the specific help on configuring your own SpellNet, or the effects, or the like.