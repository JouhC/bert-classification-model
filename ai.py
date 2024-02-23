# Step 0: Import libraries
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import pandas as pd
import tensorflow as tf

# Step 1: Load fine-tuned model, tokenizer, and encoder
MODEL = TFBertForSequenceClassification.from_pretrained("./vupico-model-pretrained")
TOKENIZER = BertTokenizer.from_pretrained("./vupico-model-pretrained")
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.classes_ = np.load('./vupico-model-pretrained/classes.npy', allow_pickle=True)

class Classifier:
    """Classifer for Object Component Classifier"""
    # Step 2: Confidence Level Estimation with Threshold
    def predict_with_confidence_threshold(text, threshold=0.06):
        """_summary_

        Args:
            text (str): this the object to identify
            threshold (float, optional): this is the min. threshold to list down all the output from the model.
                                         Defaults to 0.06.

        Returns:
            classes_above_threshold (list): these are the components identified.
            probabilities_above_threshold (list): these are the probabilities or inference score.
        """        
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
        """_summary_

        Args:
            object (str): the object we want to classify its components.

        Returns:
            output (str): a summary in bullet point which composed of components and its inference score.
        """        
        predicted_classes, confidence_levels = cls.predict_with_confidence_threshold(object)
        components = LABEL_ENCODER.inverse_transform(predicted_classes)
        trimmer = lambda x: x.strip()
        compfunc = np.vectorize(trimmer)
        components = compfunc(components) if not components.size == 0 else ["N/A"]
        confidence_levels = confidence_levels if not confidence_levels.size == 0 else [0]

        output = ''.join((f" - {component} with an inference score of {'%.4f' % level}\n" for component, level in zip(components, confidence_levels)))
        return output