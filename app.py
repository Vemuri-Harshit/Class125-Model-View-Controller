from flask import Flask, jsonify, request;
from prediction import getpredictions;

app = Flask(__name__);
@app.route("/predictdigit", methods=['POST'])

def predictdata():
    image = request.files.get('digit');
    prediction = getpredictions(image);
    return jsonify({
        'prediction': prediction
    });

if __name__ == '__main__':
    app.run(debug = True);
