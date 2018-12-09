from flask import Flask,render_template
from flask_restful import Resource, Api
import red_wine_quality_prediction
import white_wine_quality_prediction

app = Flask(__name__)

api = Api(app)
api_uri = "http://127.0.0.1:5000"

class red_wines(Resource):
    def get(self):
        return {'data': red_wine_quality_prediction.main()}

class white_wines(Resource):
    def get(self):
        return {'data': white_wine_quality_prediction.main()}

api.add_resource(red_wines, '/api/v1/get_red_wine_quality_prediction')
api.add_resource(white_wines, '/api/v1/get_white_wine_quality_prediction')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/redwinequalityprediction')
def redwinequalityprediction():
    return render_template("redwinequalityprediction.html")

@app.route('/whitewinequalityprediction')
def whitewinequalityprediction():
    return render_template("whitewinequalityprediction.html")

if __name__ == "__main__":
    app.run(debug=True)