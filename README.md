# PiPotter

A raspberry pi camera wand reader, loosely based, yet inspired,  on [Rpotter](https://github.com/sean-obrien/rpotter/blob/master/rpotter.py)

# Architecture

![](https://i.imgur.com/m5g8WOs.jpg)

The short version: Using a PiNoir camera, once an IR-reflective wand is detected, a 3 seconds movement pattern is recorded. That pattern, a sigil from now on, is passed trough a pre-trained small neural network (SpellNet) and the identified spell is sent to a controller that runs an effect, that may be lights (using the PythonMilight interface) or sound (mplayer)
The detection does looks like this, roughly:

![](https://i.imgur.com/uQiWFLs.png)

