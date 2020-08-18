from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np




class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rf_classifier=RandomForestClassifier()
        self.xgb = XGBClassifier(objective='multi:softprob',n_jobs=-1)

    def get_best_params_for_rf(self,train_x,train_y):
        """
        Method Name: get_best_params_for_rf
        Description: get the parameters for the Random Forest Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The  random forest model with the best parameters
        On Failure: Raise Exception



                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [50,100,200,300,500],
                          "criterion":  ['gini','entropy'],
                          'max_features' : ['auto','sqrt','log2'] ,
                          'max_depth' : [2,5,6,8],
                          'random_state': [42,50] }

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rf_classifier, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(np.array(train_x), np.array(train_y))

            #extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.criterion = self.grid.best_params_['criterion']
            self.max_features = self.grid.best_params_['max_features']
            self.max_depth = self.grid.best_params_['max_depth']
            self.random_state = self.grid.best_params_['random_state']




            #creating a new model with the best parameters
            self.rf_classifier = RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,max_features=self.max_features,max_depth=self.max_depth,random_state=self.random_state)
            # training the mew model
            self.rf_classifier.fit(np.array(train_x), np.array(train_y))
            self.logger_object.log(self.file_object,'Random forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random forest method of the Model_Finder class')

            return self.rf_classifier
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_for_rf method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object, 'randonm forest training  failed. Exited the get_best_params_for_rf method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception



                                """
        self.logger_object.log(self.file_object,'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                "n_estimators": [50,100,500,1000,2000], "criterion": ['gini','entropy'],
                               "max_depth": range(8,10,1)

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='multi:softprob'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(np.array(train_x), np.array(train_y))

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(criterion=self.criterion, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1 )
            # training the mew model
            self.xgb.fit(np.array(train_x), np.array(train_y))
            self.logger_object.log(self.file_object,'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def multiclass_roc_auc_score(self,y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception



                                        """
        self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(np.array(test_x)) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = self.multiclass_roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Random Forest
            self.rf=self.get_best_params_for_rf(train_x,train_y)
            self.prediction_rf=self.rf.predict(np.array(test_x)) # prediction using the Random Forest  Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.rf_score = accuracy_score(test_y,self.prediction_rf)
                self.logger_object.log(self.file_object, 'Accuracy for Random forest:' + str(self.rf_score))
            else:
                self.rf_score = self.multiclass_roc_auc_score(test_y, self.prediction_rf) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for Random Forest:' + str(self.rf_score))

            #comparing the two models
            if(self.rf_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost,self.xgboost_score
            else:
                return 'RF',self.rf_classifier,self.rf_score

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

