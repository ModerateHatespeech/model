# Machine Learning Models
In the interest of research and contribution into the performance of our language models, we release previous versions of our models for public use.

## License
All models are licensed under the GPL-v3 license, and released on an as-is basis. We make no guarantee of the accuracy or performance of these models, and you implement them under the expectation that these models can potentially contain biases as a result of the training process.

## Download

```bash
wget https://moderatehatespeech.com/research/models/model_v7.h5 -O model_v7.h5
```

## Usage
The following weights are provided as TF weights. You should install requirements via:
```pip3 install -r requirements.txt``` 
To load the model, import the necessary classes and load the model weights:
```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model = TFAutoModelForSequenceClassification.from_pretrained("roberta-large")
tokenizer = AutoTokenizer.from_pretrained('roberta-large')

model.load_weights("model_v7.h5")
```

To run a inference, you can use the default tokenizer:

```python
tokens = tokenizer(string, padding=True,truncation=True, return_tensors='tf', max_length=256)
logits=model.predict(dict(tokens))
print(tf.nn.softmax(logits.logits, axis=1).numpy())
```

## Training
Information on our training process can be found on our [website](https://moderatehatespeech.com/framework/)
