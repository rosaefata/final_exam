from flask import Flask, request, jsonify
import numpy
import skin_prediction
import importlib
importlib.reload(skin_prediction) # For reloading after making changes


app = Flask(__name__)

def percentage(num):
    return round(num*100)


@app.route('/skin-type', methods=['POST'])
def query_example():

    request_body = request.get_json()

    oiliness_ans = numpy.asarray(request_body['oiliness'])
    oiliness_ans = oiliness_ans.reshape(1, 8)

    oiliness, ol_score = skin_prediction.get_oiliness_skin_type(oiliness_ans)

    tightness_ans = numpy.asarray(request_body['tightness'])
    tightness_ans = tightness_ans.reshape(1, 13)
    tightness, ti_score = skin_prediction.get_tightness_skin_type(tightness_ans)

    sensitivity_ans = numpy.asarray(request_body['sensitivity'])
    sensitivity_ans = sensitivity_ans.reshape(1, 13)
    sensitivity, sen_score = skin_prediction.get_sensitivity_skin_type(sensitivity_ans)

    pigmentation_ans = numpy.asarray(request_body['pigmentation'])
    pigmentation_ans = pigmentation_ans.reshape(1, 7)
    pigmentation, pig_score = skin_prediction.get_pigmentation_skin_type(pigmentation_ans)

    result = oiliness+sensitivity+pigmentation+tightness
    skin_type = {'skin-type':result,'oiliness':percentage(ol_score), 'tightness':percentage(ti_score), 'sensitivity':percentage(sen_score), 'pigmentation':percentage(pig_score)}

    return jsonify(skin_type)


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
