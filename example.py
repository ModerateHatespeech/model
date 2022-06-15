from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model = TFAutoModelForSequenceClassification.from_pretrained("roberta-large")
tokenizer = AutoTokenizer.from_pretrained('roberta-large')

model.load_weights("model_v7.h5")

def predict(string):
  tokens = tokenizer(string, padding=True,truncation=True, return_tensors='tf', max_length=256)
  logits=model.predict(dict(tokens))
  return tf.nn.softmax(logits.logits, axis=1).numpy()

print(predict("Testing, 123!"))
