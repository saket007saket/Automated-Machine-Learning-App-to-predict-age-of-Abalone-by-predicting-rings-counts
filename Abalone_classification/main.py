from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
import pandas as pd
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation  # here we are calling  this class train_validation method from our file
# traing_validatio_insertion.py file
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])  # "/" since its empty route then its homepage application
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = pred_validation(path)  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization
            pred_input_data = pd.read_csv("Prediction_FileFromDB/inputFile.Csv")

            # predicting for dataset present in database
            path,result = pred.predictionFromModel()
            X = pd.concat([pred_input_data,pd.DataFrame(result)],axis=1,sort=False)
            print(X)
            return Response("Prediction File created at %s!!!" %path +"  "+  "prediction results are given below %s" %X.to_html())

        elif request.form is not None:
            path = request.form['filepath']

            pred_val = pred_validation(path)  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization
            pred_data = pd.read_csv("Prediction_FileFromDB/inputFile.Csv")
            # predicting for dataset present in database
            path ,result= pred.predictionFromModel()
            X = pd.concat([pred_data, pd.DataFrame(result)], axis=1, sort=False)
            return Response("Prediction File created at %s!!!" %path + "   "+"prediction results are given below %s" %X.to_html())

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json is not None:  # 'folderPath' is nothing but path of folder we provide where trainign data set file is placed
            path = request.json['filepath']
            train_valObj = train_validation(path)  # object initialization for class train_Validation(we r creating a instance of class train_validation
            # as train_valObj

            train_valObj.train_validation()  # calling the training_validation function(from hat object calling taining_Validation function)

            trainModelObj = trainModel()  # object initialization
            summary_of_training = trainModelObj.trainingModel()  # training the model for the files in the table
            print(summary_of_training)
            return Response("result of training  %s!!!" %summary_of_training.to_html() )


        elif request.form is not None:
            path = request.form['filepath']
            train_valObj = train_validation(path)  # object initialization for class train_Validation(we r creating a instance of class train_validation
            # as train_valObj

            train_valObj.train_validation()  # calling the training_validation function(from hat object calling taining_Validation function)

            trainModelObj = trainModel()  # object initialization
            summary_of_training = trainModelObj.trainingModel()  # training the model for the files in the table
            print(summary_of_training)
            return Response("result of training  %s!!!" %summary_of_training.to_html())

            
    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)


port = int(os.getenv("PORT", 5001))
if __name__ == "__main__":
    app.run(port=port, debug=True)

    #app.run(port=8000,host='0.0.0.0')
