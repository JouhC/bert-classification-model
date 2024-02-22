# Step 0: Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import math


# Step 1: Load fine-tuned model, tokenizer, and encoder
MODEL = TFBertForSequenceClassification.from_pretrained("./vupico-model-pretrained")
TOKENIZER = BertTokenizer.from_pretrained("./vupico-model-pretrained")
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.classes_ = np.load('./vupico-model-pretrained/classes.npy', allow_pickle=True)

class Classifier:
    # Step 2: Confidence Level Estimation with Threshold
    def predict_with_confidence_threshold(text, threshold=0.06):
        encoded_text = TOKENIZER(text, padding=True, truncation=True, return_tensors="tf")
        logits = MODEL.predict(dict(encoded_text))[0]
        probabilities = tf.nn.softmax(logits, axis=1)
        
        # Get the indices of probabilities greater than the threshold
        indices_above_threshold = tf.where(probabilities > threshold)
        
        # Get the classes and corresponding probabilities above the threshold
        classes_above_threshold = indices_above_threshold[:, 1].numpy()
        probabilities_above_threshold = probabilities.numpy()[indices_above_threshold[:, 0], indices_above_threshold[:, 1]]
        
        return classes_above_threshold, probabilities_above_threshold

    @classmethod
    def classify(cls, object: str):
        predicted_classes, confidence_levels = cls.predict_with_confidence_threshold(object)
        components = LABEL_ENCODER.inverse_transform(predicted_classes)
        trimmer = lambda x: x.strip()
        compfunc = np.vectorize(trimmer)
        components = compfunc(components)

        output = ''.join((f" - {component} with an inference score of {'%.4f' % level}\n" for component, level in zip(components, confidence_levels)))
        return output