# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import numpy as np
import pandas as pd

#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()


            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data=preprocessor.remove_columns(data,[ ]) # remove the column as it doesn't contribute to prediction.
            data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation

            # since in our target column we have multi no labels and it is higly imbalance so i m grouping them using below preprocessor method

            data = preprocessor.grouping_values_of_target(data)

            # create separate two data frames one on which we will perform cluster  and other is attached after performing cluster
            X, Y =preprocessor.separate_data_frame(data,label_column_name=['Rings','Sex'])


            """ Applying the clustering approach"""

            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer) # object initialization.
            number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X = pd.concat([X, Y], axis=1, sort=False)

            # encode categorical data
            X = preprocessor.encode_categorical_columns(X)

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
            df = pd.DataFrame(columns = ['Cluster_No', 'Best_Model_Name', 'Roc_Auc_score'])


            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Rings','Cluster'],axis=1)
                cluster_label= cluster_data['Rings']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=100)


                model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

                #getting the best model for each of the clusters
                best_model_name,best_model,Roc_Auc_score=model_finder.get_best_model(x_train,y_train,x_test,y_test)

                #saving the best model to the directory.
                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))
                df = df.append({'Cluster_No' : i,  'Best_Model_Name' : best_model_name+str(i),'Roc_Auc_score':Roc_Auc_score},  ignore_index = True)

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()
            return df

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception