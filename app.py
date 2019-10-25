from flask import Flask
from resources.knn import Neighbours
from resources.svm import SVM
from flask_restful import (Api)

app = Flask(__name__)
api = Api(app)


api.add_resource(Neighbours, '/knn')
api.add_resource(SVM, '/svm')

app.run(port=5000)