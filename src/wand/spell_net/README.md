# SpellNet

Just a very simple keras based convoluted neural network, MNIST based, that reads the wand movements.

Please do refer to the corpus_generator and spell_net Python ebooks

## Usage

```python
from wand.spell_net.model import SpellNet
m = SpellNet()
m.predict(a_32x32x3_normalized_to_127.5_image)
```
The result of the aforementioned command should be a dictionary similar to:

```python
{'noctis': 0.99863762, 'silencio': 3.9032732e-29, 'locomotor': 0.0, 'meteojinx': 3.7186632e-23, 'arresto_momentum': 0.0, 'lumos': 0.0013623675}
```

Both the class names and the path to get the models are set in the core/config module.

