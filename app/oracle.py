import sys
import os
import logging
import json
from urllib.request import ProxyBasicAuthHandler
import numpy as np
from tensorflow.keras import models
import tensorflow as tf
# import tflite_runtime.interpreter as tflite

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from helper import textutils

class Oracle:

    logger = logging.getLogger(__name__)
    def __init__(self):
        self._kMODEL_PATH = os.path.join(parent_dir, "train", "models")
        self._config = None
        self._loaded_models = None
        self._labels = None

        self._load_config()
        self._load_user_guide()
        self._get_model_opts()
 

    def _load_config(self):
        config_file = os.path.join(self._kMODEL_PATH, 'configure.json')
        with open(os.path.join(config_file), 'r', encoding='utf-8') as fp:
            self._config = json.load(fp)

        self._labels = dict()

        for k, v in self._config.get('labels').items():
            v = int(v)
            if v in self._labels:
                self._labels[v] = f'{self._labels[v]}/{k}'
            else:
                self._labels[v] = k


    def _get_model_opts(self):
        self._loaded_models = dict()

        if not self._user_guide:
            self._load_user_guide()

        model_user_guide = self._user_guide.get('models', {})


        for file in os.listdir(self._kMODEL_PATH):
            if file.endswith('.h5'):
                name = file.replace('.h5', '')
                path = os.path.join(self._kMODEL_PATH, file)
                model = models.load_model(path)
                probability_model = tf.keras.Sequential([model,
                                                         tf.keras.layers.Softmax()])
                # model = tflite.Interpreter(path)
                # probability_model = model


                self._loaded_models[name] = {
                    'model': models.load_model(path),
                    'prob_model': probability_model,
                    'name': model_user_guide.get(name).get('name'),
                    'desc': model_user_guide.get(name).get('desc'),
                    'path': path,
                }

        return self._loaded_models

    def _load_user_guide(self):
        kUSER_GUIDE = 'user_guide.json'
        with open(kUSER_GUIDE, 'r') as fp:
            self._user_guide = json.load(fp)


    def get_model_opts(self):
        model_opts = dict()
        for name, model in self._loaded_models.items():
            model_opts[name] = {
                'name': model.get('name'),
                'desc': model.get('desc'),
            }
        return model_opts

    def get_user_guide(self, label):
        return self._user_guide.get('labels').get(label)

    def label_predictions(self, predictions):
        labeled_pred = dict()
        for i, val in enumerate(predictions):
            labeled_pred[self._labels[i]] = val

        return labeled_pred

    def predict(self, model_name, text):

        if model_name not in self._loaded_models:
            # Select the first model (default)
            model_name = list(self._loaded_models.keys())[0]

        model = self._loaded_models.get(model_name)

        sanitized_text = textutils.sanitize_text(text)
        tokenized_text = textutils.tokenize(sanitized_text, clean=True)

        # Choose the right
        textrep_array = None
        if model_name.endswith('_bow'):
            textrep_array = textutils.bow_array_from_text(sanitized_text,
                                                          self._config['word_index'],
                                                          ignore_err=True)
        elif model_name.endswith('_tfidf'):
            textrep_array = textutils.tfidf_array_from_text(sanitized_text,
                                                            self._config['word_index'],
                                                            self._config['idf_values'],
                                                            ignore_err=True)

        elif model_name.endswith('_seqpad'):
            textrep_array = textutils.padded_seq_from_text(sanitized_text,
                                                        self._config['word_index'],
                                                        self._config['max_length'],
                                                        ignore_err=True)
        else:
            raise RuntimeError(f'Unknown text representation (model={model_name})')

        predictions = model.get('prob_model').predict(np.array([textrep_array]))
        predictions = predictions[0]
        predict = np.argmax(predictions)
        self.logger.info(f'Prediction (model={model_name}, predict={predict})')
        predict = self._labels[predict]

        return {
            'text': text,
            'model': model.get('name'),
            'model_key': model_name,
            'sanitized_text': sanitized_text,
            'tokenized_text': tokenized_text,
            'predictions': predictions,
            'predict': predict,
        }
