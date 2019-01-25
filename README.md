# Doctor-Predictor
Doctor Predictor is GUI application to train data sets on sklearn classifiers. It trains models through 4 sklearn classifiers(Random Forest, Linear Discriminant Analysis model, Gaussian model and Extra Tree Classifier).
The prediction is made through the model which gives maximum accuracy, with extensive features. Hopefully it may help a lot who are new to machine learning.

DataFile Directory.
pd_speech_features.csv is the dataset for parkinson disease, parkinson_patient.csv is the sample data for prediction.
blood.txt is the dataset for blood transfusion, bloodPatient.txt is the sample data for prediction.
BreastCancerData.txt is the data for breast cancer patients, BreastCancerPatient.txt is the sample data for prediction
key.txt contains information to upload data to the firebase server.

change the location of the following variables (as per your DataFile Locationa) in the doctorPredictor.py file.

self.FILE = 'C:/Users/AA/Desktop/Doctor Peredictor/DataFile/'

self.KEYS = 'C:/Users/AA/Desktop/Doctor Peredictor/DataFile/keys.txt'



modules used:
tkinter
firebase
time
pandas
numpy
sklearn
seaborn
mpl_toolkits.mplot3d
matplotlib

Data set are collected from UCI respiratory.
*link for breast cancer data set: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
link for blood transfusion data set: https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/
link for parkinson data set: https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification#
*Missing values exist.


