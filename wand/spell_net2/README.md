# SpellNet2

Just a very simple EfficientNetB0 that reads the wand movements.

Please do refer to the corpus_generator for how to build the corpus. For training, resort to  [TFLite Model Maker Docker Image](https://www.data-mining.co.nz/news/2021-10-06-tflite-model-maker/)

## Usage

```python
from wand.spell_net2.model import SpellNet
m = SpellNet()
m.predict(a_224x224x3_image_in_numpy_format)
```
The result of the aforementioned command should be a dictionary similar to:

```python
{'alohomora': 0.99863762, 'arresto_momentum': 3.9032732e-29, 'incendio': 0.0, 'locomotor': 3.7186632e-23, 'lumos': 0.0, 'revelio': 0.0013623675}
```

Both the class names and the path to get the models are set in the core/config module.

