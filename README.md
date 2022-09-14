# Machine Learning Models
In the interest of research and contribution into the performance of our language models, we release previous versions of our models for public use.

## License
All models are licensed under the GPL-v3 license, and released on an as-is basis. We make no guarantee of the accuracy or performance of these models, and you implement them under the expectation that these models can potentially contain biases as a result of the training process.

## Download

```bash
wget https://moderatehatespeech.com/research/models/pyt_v2.pkl -O pyt_v2.pkl
```

## Usage
The following weights are provided as FastAI (Pytorch) pickled weights. You should install requirements via:
```pip3 install -r requirements.txt``` 
To load the model, import the necessary classes from the mhs_imports.py file and load the model:
```python
from mhs_imports import *

model = load_learner("./", "pyt_v2.pkl")
```

To run a inference:

```python
model.predict("Testing, 123")
```

## Training
Information on our training process can be found on our [website](https://moderatehatespeech.com/framework/)

