import os
import sys
import logging
from flask import Flask, render_template, request, jsonify

from oracle import Oracle
from feedback import Feedback

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

oracle = Oracle()
feedback_collector = Feedback()
feedback_collector.init_db()

models = oracle.get_model_opts()

def predict(model, sms_text):

    def help_pred_explain(pred_result):

        pred_result['predictions'] = oracle.label_predictions(pred_result.get('predictions'))
        return {
            **pred_result,
            **oracle.get_user_guide(pred_result.get('predict')),
        }

    pred_result = oracle.predict(model, sms_text)
    pred_result = help_pred_explain(pred_result)

    for key, val in pred_result['predictions'].items():
        pred_result['predictions'][key] = float(val)

    logger.info(pred_result)

    return pred_result


@app.route('/predict', methods=['POST'])
def predict_API():
    data     = request.get_json()
    sms_text = data.get('sms_text', None)
    model    = data.get('model', None)
    response = {
        'sms_text': sms_text,
        'model': model,
        'pred_result': predict(model, sms_text),
    }

    return jsonify(response)

    # curl --header "Content-Type: application/json" \
    #      --request POST \
    #      --data '{"sms_text": "Hello, world!", "model": "dense"}' \
    #      http://localhost:5000/predict

@app.route('/feedback', methods=['POST'])
def feedback_API():
    data = request.get_json()

    # Save feedbacks for re-train and improve prediction models
    res = feedback_collector.save(
        sms_text=data.get('sms_text', None),
        model_key=data.get('model_key', None),
        pred_class=data.get('pred_class', None),
        feedback_positive=data.get('feedback_positive', None),
        feedback_class=data.get('feedback_class', None),
        feedback_text=data.get('feedback_text', None),
    )
    return jsonify({ 'successful': res })

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        data = request.form.to_dict()
        if data.get('sms_text'):
            pred_result = predict(data['model'], data['sms_text'])
            return render_template('index.html',
                                   models=models,
                                   pred_result=pred_result,
                                   selected=data['model'],
                                   sms_text=data['sms_text']) 

    return render_template('index.html',
                           models=models,
                           pred_result=None,
                           selected=None,
                           sms_text=None)
