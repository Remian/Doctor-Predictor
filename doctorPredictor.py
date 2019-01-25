from tkinter import *
from firebase import firebase
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as py
from pandas.plotting import parallel_coordinates


class guiML(Frame):

    def __init__(self, master, *args, **kwargs):

        Frame.__init__(self, *args, **kwargs)

        frame = Frame(master)
        master.wm_iconbitmap('logo.ico')
        master.wm_title('Doctor Predictor')


        self.diagnosis = StringVar()
        self.diagnosis.set("no diagnosis")
        self.statusVar = StringVar()
        self.statusVar.set("not trained")
        self.algoName = StringVar()
        self.algoName.set("not available")
        self.checkButtonState = IntVar()

        self.title = Label(frame, text="DOCTOR PREDICTOR")
        self.trainingFile_label = Label(frame, text="Training Data File")
        self.status_title = Label(frame, text="Status")
        self.status_var = Label(frame, textvariable=self.statusVar)
        self.predictionFile_label = Label(frame, text="Patient Data")
        self.diagnosis_label = Label(frame, text="Diagnosis/class")
        self.result_label = Label(frame, textvariable=self.diagnosis)
        self.algoUsed_label = Label(frame, text="Algorithm used for prediction")
        self.usedAlgo_label = Label(frame, textvariable=self.algoName)
        self.trainingFile_entry = Entry(frame)
        self.predictionFile_entry = Entry(frame)
        self.runButton = Button(frame, text="Train", bg="black", fg="white", command=self.trainAndPredict)
        self.predictButton = Button(frame, text="predict", bg="black", fg="white", command=self.predict)
        self.checkButton = Checkbutton(frame, text="upload data to firebase server", variable=self.checkButtonState)
        self.quitButton = Button(frame, text="QUIT", command=frame.quit)




        frame.pack(fill="both",expand=True)



        self.title.grid(row=0, column=1)
        self.trainingFile_label.grid(row=1, column=0, sticky=W)
        self.trainingFile_entry.grid(row=1, column=2, sticky=E)


        self.checkButton.grid(columnspan=3)
        self.runButton.grid(row=4, column=2, sticky=E)
        self.status_title.grid(row=5, column=0, sticky=W)
        self.status_var.grid(row=5, column=2, sticky=W)
        self.predictionFile_label.grid(row=6, column=0, sticky=W)
        self.predictionFile_entry.grid(row=6, column=2, sticky=E)
        self.predictButton.grid(row=7, column=2, sticky=E)
        self.diagnosis_label.grid(row=8, column=0, sticky=W)
        self.result_label.grid(row=8, column=2, sticky=W)
        self.algoUsed_label.grid(row=9, column=0, sticky=W)
        self.usedAlgo_label.grid(row=9, column=2, sticky=W)
        self.quitButton.grid(row=10, column=2, sticky=E)


        self.parkinsonForestClassifier = RandomForestClassifier(n_estimators=1000)
        self.ParkinsonLinearClassifier = LinearDiscriminantAnalysis()
        self.ParkinsonGbClassifier = GaussianNB()
        self.ParkinsonExtraTreeClassifier = ExtraTreesClassifier(n_estimators=1000)

        self.scoreCount = dict()
        self.timeCount = dict()
        self.classifiers = {self.parkinsonForestClassifier: 'RandomForestClassifier', self.ParkinsonLinearClassifier: 'LinearDiscriminantAnalysis', self.ParkinsonGbClassifier: 'GaussianNB', self.ParkinsonExtraTreeClassifier: 'ExtraTreesClassifier'}

        self.FILE = 'C:/Users/AA/Desktop/Doctor Peredictor/DataFile/'
        self.KEYS = 'C:/Users/AA/Desktop/Doctor Peredictor/DataFile/keys.txt'
        self.APP_DATABASE_URL = 'https://ecg-1-e1a65.firebaseio.com/'

        #addins

        self.featureDic = dict()

        self.menu = Menu(master)
        master.config(menu=self.menu)

        self.infoMenu = Menu(self.menu)
        self.menu.add_cascade(label='information', menu=self.infoMenu)
        self.infoMenu.add_command(label="patient information", command=self.patInfo)
        self.infoMenu.add_separator()
        self.infoMenu.add_command(label="features", command=self.fetureInfo)
        self.infoMenu.add_separator()
        self.infoMenu.add_command(label="model information", command=self.modelInfo)




        self.plotMenu = Menu(self.menu)
        self.menu.add_cascade(label='plotter', menu=self.plotMenu)
        self.plotMenu.add_command(label="2D plotter", command=self.plotterTwo)
        self.plotMenu.add_separator()
        self.plotMenu.add_command(label="3D plotter", command=self.plotterThree)

        window = Toplevel(self)
        title = Label(window, text='3D plotter')
        title.pack()

        for i in range(0, 11):
            frame.grid_rowconfigure(i, weight=1)

        for i in range(0, 3):
            frame.grid_columnconfigure(i, weight=1)

    def patInfo(self):

        self.patInfoWindow = Toplevel(self)
        self.patInfoWindow.wm_iconbitmap('logo.ico')
        self.patInfoWindow.wm_title('Doctor Predictor')

        title = Label(self.patInfoWindow, text='Patient Information')
        idtext = Text(self.patInfoWindow, height=5, width=15)
        idtextBar = Scrollbar(self.patInfoWindow)
        toShow_label = Label(self.patInfoWindow, text='enter patient id')
        self.toShow_Patentry = Entry(self.patInfoWindow)
        self.toShow_Patchbutt = Checkbutton(self.patInfoWindow, text='click to show', command=self.patInfoPrint)

        idtext.insert(END, "available IDs: \n")

        for i in self.idlist:

            idtext.insert(END, str(i) + "\n")


        idtextBar.config(command=idtext.yview)
        idtext.config(yscrollcommand=idtextBar.set)

        title.grid(row=0, column=1)
        idtext.grid(row=1, rowspan=2, columnspan=3)
        idtextBar.grid(row=1, column=2, rowspan=2, columnspan=1, sticky=N+S+W)

        toShow_label.grid(row=3, column=0)
        self.toShow_Patentry.grid(row=3, column=1)
        self.toShow_Patchbutt.grid(row=3, column=2)

        for i in range(0, 4):
            self.patInfoWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 3):
            self.patInfoWindow.grid_columnconfigure(i, weight=1)


    def patInfoPrint(self):

        patID = int(self.toShow_Patentry.get())
        patText = Text(self.patInfoWindow, height=5, width=25)

        indexID = self.idlist.index(patID)
        textvar = self.data.ix[indexID]

        patText.insert(END, "Patient information\n")
        patText.insert(END, textvar)
        patbar = Scrollbar(self.patInfoWindow)
        patbar.config(command=patText.yview)
        patText.config(yscrollcommand=patbar.set)

        patText.grid(row=4, rowspan=3, columnspan=3)
        patbar.grid(row=4, column=3, rowspan=3, columnspan=1, sticky=N + S + W)

        for i in range(0, 5):
            self.patInfoWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 4):
            self.patInfoWindow.grid_columnconfigure(i, weight=1)

    def fetureInfo(self):

        self.featureInfoWindow = Toplevel(self)
        self.featureInfoWindow.wm_iconbitmap('logo.ico')
        self.featureInfoWindow.wm_title('Doctor Predictor')

        title = Label(self.featureInfoWindow, text='FEATURES INFORMATION')
        tt = Text(self.featureInfoWindow, height=5, width=15)
        scbar = Scrollbar(self.featureInfoWindow)
        tt.insert(END, "USED FEATURES\n")

        for i in self.featuresNames:
            tt.insert(END, i+"\n")


        toShow = Label(self.featureInfoWindow, text='Feature value to show')
        self.toShow_entry = Entry(self.featureInfoWindow)
        toShow_state = IntVar()
        toShow_state.set(0)
        toShow_chbutt = Checkbutton(self.featureInfoWindow, text="click to show", variable=toShow_state,
                                    command=self.fetureInfoPrint)

        scbar.config(command=tt.yview)
        tt.config(yscrollcommand=scbar.set)

        title.grid(row=0, column=1, sticky=W)
        tt.grid(row=1, rowspan=2, columnspan=3)
        scbar.grid(row=1, column=2, rowspan=2, columnspan=1, sticky=N + S + W)

        toShow.grid(row=3, column=0)
        self.toShow_entry.grid(row=3, column=1)
        toShow_chbutt.grid(row=3, column=2)

        for i in range(0, 4):
            self.featureInfoWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 2):
            self.featureInfoWindow.grid_columnconfigure(i, weight=1)



    def fetureInfoPrint(self):

        fetName = self.toShow_entry.get()
        fettext = Text(self.featureInfoWindow, height=5, width=15)
        textvar = self.featureDic[fetName]
        fettext.insert(END, "values\n")
        fettext.insert(END, textvar)
        fetbar = Scrollbar(self.featureInfoWindow)
        fetbar.config(command=fettext.yview)
        fettext.config(yscrollcommand=fetbar.set)

        fettext.grid(row=4, rowspan=3, columnspan=1)
        fetbar.grid(row=4, column=1, rowspan=3, columnspan=1, sticky=N + S + W)

        for i in range(0, 5):
            self.featureInfoWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 3):
            self.featureInfoWindow.grid_columnconfigure(i, weight=1)

    def modelInfo(self):

        modelInfoWindow = Toplevel(self)
        modelInfoWindow.wm_iconbitmap('logo.ico')
        modelInfoWindow.wm_title('Doctor Predictor')

        #variables

        randfrorestScore = self.scoreCount[self.parkinsonForestClassifier]*100
        randforestTime = self.timeCount[self.parkinsonForestClassifier]
        randfrorestScoreText = str(randfrorestScore)+" "+"%"
        randforestTimeText = str(randforestTime)+" "+"sec"
        randfrorestScoreVar = StringVar()
        randfrorestScoreVar.set(randfrorestScoreText)
        randforestTimeVar = StringVar()
        randforestTimeVar.set(randforestTimeText)

        linearScore = self.scoreCount[self.ParkinsonLinearClassifier] * 100
        linearTime = self.timeCount[self.ParkinsonLinearClassifier]
        linearScoreText = str(linearScore) + " " + "%"
        linearTimeText = str(linearTime) + " " + "sec"
        linearScoreVar = StringVar()
        linearScoreVar.set(linearScoreText)
        linearTimeVar = StringVar()
        linearTimeVar.set(linearTimeText)

        gnbScore = self.scoreCount[self.ParkinsonGbClassifier] * 100
        gnbTime = self.timeCount[self.ParkinsonGbClassifier]
        gnbScoreText = str(gnbScore) + " " + "%"
        gnbTimeText = str(gnbTime) + " " + "sec"
        gnbScoreVar = StringVar()
        gnbScoreVar.set(gnbScoreText)
        gnbTimeVar = StringVar()
        gnbTimeVar.set(gnbTimeText)

        extreeScore = self.scoreCount[self.ParkinsonExtraTreeClassifier] * 100
        extreeTime = self.timeCount[self.ParkinsonExtraTreeClassifier]
        extreeScoreText = str(extreeScore) + " " + "%"
        extreeTimeText = str(extreeTime) + " " + "sec"
        extreeScoreVar = StringVar()
        extreeScoreVar.set(extreeScoreText)
        extreeTimeVar = StringVar()
        extreeTimeVar.set(extreeTimeText)


        #labels

        labelRandforest = Label(modelInfoWindow, text="Random Forest Classifier")
        labelRandforestAccuracy = Label(modelInfoWindow, text="Model accuracy : ")
        labelRandforestScore = Label(modelInfoWindow, textvariable=randfrorestScoreVar)
        labelRandforestRunTime = Label(modelInfoWindow, text="run time : ")
        labelRandforestTime = Label(modelInfoWindow, textvariable=randforestTimeVar)

        labelLinear = Label(modelInfoWindow, text="Linear Discriminant Analysis")
        labelLinearAccuracy = Label(modelInfoWindow, text="Model accuracy : ")
        labelLinearScore = Label(modelInfoWindow, textvariable=linearScoreVar)
        labelLinearRunTime = Label(modelInfoWindow, text="run time : ")
        labelLinearTime = Label(modelInfoWindow, textvariable=linearTimeVar)

        labelgnb = Label(modelInfoWindow, text="Gausian Naive Bias")
        labelgnbAccuracy = Label(modelInfoWindow, text="Model accuracy : ")
        labelgnbScore = Label(modelInfoWindow, textvariable=gnbScoreVar)
        labelgnbRunTime = Label(modelInfoWindow, text="run time : ")
        labelgnbTime = Label(modelInfoWindow, textvariable=gnbTimeVar)

        labelExtree = Label(modelInfoWindow, text="Extra Tree Classifier")
        labelExtreeAccuracy = Label(modelInfoWindow, text="Model accuracy : ")
        labelExtreeScore = Label(modelInfoWindow, textvariable=extreeScoreVar)
        labelExtreeRunTime = Label(modelInfoWindow, text="run time : ")
        labelExtreeTime = Label(modelInfoWindow, textvariable=extreeTimeVar)

        #grid

        labelRandforest.grid(row = 0, column=1)
        labelRandforestAccuracy.grid(row=1, column=0, sticky=W)
        labelRandforestScore.grid(row=1, column=2, sticky=W)
        labelRandforestRunTime.grid(row=2, column=0, sticky=W)
        labelRandforestTime.grid(row=2, column=2, sticky=W)

        labelLinear.grid(row=3, column=1)
        labelLinearAccuracy.grid(row=4, column=0, sticky=W)
        labelLinearScore.grid(row=4, column=2, sticky=W)
        labelLinearRunTime.grid(row=5, column=0, sticky=W)
        labelLinearTime.grid(row=5, column=2, sticky=W)

        labelgnb.grid(row=6, column=1)
        labelgnbAccuracy.grid(row=7, column=0, sticky=W)
        labelgnbScore.grid(row=7, column=2, sticky=W)
        labelgnbRunTime.grid(row=8, column=0, sticky=W)
        labelgnbTime.grid(row=8, column=2, sticky=W)

        labelExtree.grid(row=9, column=1)
        labelExtreeAccuracy.grid(row=10, column=0, sticky=W)
        labelExtreeScore.grid(row=10, column=2, sticky=W)
        labelExtreeRunTime.grid(row=11, column=0, sticky=W)
        labelExtreeTime.grid(row=11, column=2, sticky=W)

        for i in range(0, 4):
            modelInfoWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 2):
            modelInfoWindow.grid_columnconfigure(i, weight=1)

    def dataFrameSet(self):

        dataset = self.data

        dataset = dataset.drop(['id'], axis=1)

        header = list(dataset)

        self.Header = header

        lastIndex = len(header) - 1

        featureHeader = list()

        scale = MinMaxScaler()

        for i in header:

            if i is header[lastIndex]:

                label = np.array(dataset[i])


            else:

                dataset[i] = dataset[i].replace('?', np.nan)
                dataset[i] = dataset[i].fillna(0)
                dataset[i] = dataset[i].astype(int)

                vars()[i] = np.array(dataset[i])
                vars()[i] = vars()[i].reshape(-1, 1)

                vars()[i] = scale.fit_transform(vars()[i])

                featureHeader.append(i)

        features = np.array([])

        for i in featureHeader:

            if i is featureHeader[0]:

                features = vars()[i]

            else:

                features = np.concatenate((features, vars()[i]), axis=1)

        dataFrame = np.concatenate((features, label.reshape(-1, 1)), axis=1)

        index = list()

        self.mainFrame = pd.DataFrame()

        for i in range(0, len(dataFrame)):
            index.append(i)

        for i in range(0, len(header)):
            decoy = dataFrame[:, i]
            decoy = decoy.tolist()

            frame = pd.DataFrame({header[i]: decoy}, index=index)
            self.mainFrame = pd.concat([self.mainFrame, frame], axis=1)

        #print(mainFrame)




    def plotterTwo(self):

        self.plotterTwoWindow = Toplevel(self)
        self.plotterTwoWindow.wm_iconbitmap('logo.ico')
        self.plotterTwoWindow.wm_title('Doctor Predictor')


        labelTitle = Label(self.plotterTwoWindow, text='2D plotter')
        textBoxHeader = Text(self.plotterTwoWindow, height=5, width=15)

        labelScatterTitle = Label(self.plotterTwoWindow, text='Scatter Plot', bg="white", fg="green")
        labelScatterFeature = Label(self.plotterTwoWindow, text='Place features for plot (separated by coma)')
        self.entryScatterFeature = Entry(self.plotterTwoWindow)
        runScatter = Button(self.plotterTwoWindow, text="Show", bg="black", fg="white", command=self.snslm)

        labelBoxTitle = Label(self.plotterTwoWindow, text='Box Plot', bg="white", fg="green")
        labelBoxFeature = Label(self.plotterTwoWindow, text='Place features for plot (separated by coma)')
        self.entryBoxFeature = Entry(self.plotterTwoWindow)
        runBox = Button(self.plotterTwoWindow, text="Show", bg="black", fg="white", command=self.snsbox)

        labelHeatTitle = Label(self.plotterTwoWindow, text='Heat Map', bg="white", fg="green")
        textBoxHeat = Text(self.plotterTwoWindow, height=2, width=65)
        runHeat = Button(self.plotterTwoWindow, text="Show", bg="black", fg="white", command=self.snsheatmap)

        labelCordTitle = Label(self.plotterTwoWindow, text='Parallel Coordinates', bg="white", fg="green")
        textBoxCord = Text(self.plotterTwoWindow, height=2, width=65)
        runCord = Button(self.plotterTwoWindow, text="Show", bg="black", fg="white", command=self.snscord)

        labelSwapTitle = Label(self.plotterTwoWindow, text='Swamp Plot', bg="white", fg="green")
        textBoxSwamp = Text(self.plotterTwoWindow, height=3, width=65)
        runSwamp = Button(self.plotterTwoWindow, text="Show", bg="black", fg="white", command=self.snsswamp)


        labelEnd = Label(self.plotterTwoWindow, text="Plotting Scaled MainFrame Data")


        textBoxHeader.insert(END, "Headers:\n")
        for i in self.Header:
            textBoxHeader.insert(END, i+"\n")

        textBoxHeat.insert(END, "Plot Info\n")
        textBoxHeat.insert(END, "Gives an overview of how different features are interconnected.")

        textBoxCord.insert(END, "Plot info\n")
        textBoxCord.insert(END, "Gives an overview of how separable are the data classes.")

        textBoxSwamp.insert(END, "Plot info\n")
        textBoxSwamp.insert(END, "Shows extensive information about the features.\n")
        textBoxSwamp.insert(END, "Large data files may take larger run time.\n")

        labelTitle.grid(row=0, column=0, sticky=W)
        textBoxHeader.grid(row=1, rowspan=3, columnspan=3)

        labelScatterTitle.grid(row=4, column=1, sticky=W,pady=15)
        labelScatterFeature.grid(row=5, column=0, sticky=W)
        self.entryScatterFeature.grid(row=5, column=2, sticky=E)
        runScatter.grid(row=6, column=2, sticky=E)

        labelBoxTitle.grid(row=7, column=1, sticky=W, pady=15)
        labelBoxFeature.grid(row=8, column=0, sticky=W)
        self.entryBoxFeature.grid(row=8, column=2, sticky=E)
        runBox.grid(row=9, column=2, sticky=E)

        labelHeatTitle.grid(row=10, column=1, sticky=W, pady=15)
        textBoxHeat.grid(row=11, rowspan=2, columnspan=3)
        runHeat.grid(row=13, column=2, sticky=E)

        labelCordTitle.grid(row=14, column=1, sticky=W, pady=15)
        textBoxCord.grid(row=15, rowspan=2, columnspan=3)
        runCord.grid(row=17, column=2, sticky=E)

        labelSwapTitle.grid(row=18, column=1, sticky=W, pady=15)
        textBoxSwamp.grid(row=19, rowspan=3, columnspan=3)
        runSwamp.grid(row=23, column=2, sticky=E)

        labelEnd.grid(row=24, columnspan=3, pady=25)

        for i in range(0, 25):
            self.plotterTwoWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 3):
            self.plotterTwoWindow.grid_columnconfigure(i, weight=1)

    def snslm(self):

        decoy = self.entryScatterFeature.get()
        dec = decoy.split(",")

        self.dataFrameSet()

        sns.lmplot(x=dec[0], y=dec[1], hue=self.Header[-1],
                   markers=['x', 'o'],
                   fit_reg=False, data=self.mainFrame)
        py.xticks(rotation=90)

        py.show()

    def snsbox(self):

        decoy = self.entryBoxFeature.get()
        dec = decoy.split(",")

        self.dataFrameSet()

        sns.boxplot(x=dec[0], y=dec[1], hue=self.Header[-1], data=self.mainFrame)

        py.xticks(rotation=90)

        py.show()

    def snsheatmap(self):

        self.dataFrameSet()

        sns.heatmap(self.mainFrame.head(), annot=True)

        py.xticks(rotation=90)

        py.show()

    def snscord(self):

        self.dataFrameSet()

        parallel_coordinates(self.mainFrame, self.Header[-1])

        py.xticks(rotation=90)

        py.show()

    def snsswamp(self):

        self.dataFrameSet()

        sns.set(style="whitegrid", palette="muted")

        data = pd.melt(self.mainFrame, id_vars=self.Header[-1],
                       var_name="features",
                       value_name='value')

        sns.swarmplot(x="features", y="value", hue=self.Header[-1], data=data)

        py.xticks(rotation=90)

        py.show()

    def plotterThree(self):

        plotterThreeWindow = Toplevel(self)
        plotterThreeWindow.wm_iconbitmap('logo.ico')
        plotterThreeWindow.wm_title('Doctor Predictor')

        labelTitle = Label(plotterThreeWindow, text='3D PLOTTER')
        textBoxHeader = Text(plotterThreeWindow, height=5, width=15)
        labelInput = Label(plotterThreeWindow, text='input features')
        self.entryInput = Entry(plotterThreeWindow)
        labelDoc = Label(plotterThreeWindow, text='input two features separated by coma')
        runButton = Button(plotterThreeWindow, text='show', bg='black', fg='white', command= self.plotterThreeShow)

        textBoxHeader.insert(END, "Headers\n")

        for i in self.Header:
            textBoxHeader.insert(END, i + "\n")

        labelTitle.grid(row = 0, column = 0, sticky=W)
        textBoxHeader.grid(row = 1, rowspan = 3, columnspan = 3)
        labelInput.grid(row = 4, column = 0, sticky=W)
        self.entryInput.grid(row = 4, column = 2, sticky=E)
        labelDoc.grid(row = 5, column = 0, sticky=W)
        runButton.grid(row = 5, column = 2, sticky=E)

        for i in range(0, 6):
            plotterThreeWindow.grid_rowconfigure(i, weight=1)

        for i in range(0, 3):
            plotterThreeWindow.grid_columnconfigure(i, weight=1)


    def plotterThreeShow(self):

        decoy = self.entryInput.get()

        dec = decoy.split(",")

        print(decoy)
        print(type(decoy))
        print(dec)
        print(type(dec[0]))

        fig = py.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(self.mainFrame[dec[0]], self.mainFrame[dec[1]], self.mainFrame[self.Header[-1]], c='r', marker='^')
        ax.set_xlabel(dec[0], fontsize=25)
        ax.set_ylabel(dec[1], fontsize=25)
        ax.set_zlabel(self.Header[-1], fontsize=25)
        py.show()







    def upload_file(self, file_location, directory):

        fp = open(file_location)
        fp_keys = open(self.KEYS, "a")
        dat_file = ""
        for d in fp.readlines():
            dat_file += d
        ref = firebase.FirebaseApplication(self.APP_DATABASE_URL, None)
        data = {'file_name': 'data.dat',
                'data': dat_file
                }
        result = ref.post("/" + directory + "/", data)
        fp_keys.write(data['file_name'] + " " + result['name'] + '\n')




    def trainAndPredict(self):

        chButtonLogic = self.checkButtonState.get()
        trainingFile = self.trainingFile_entry.get()
        FILE_loc = self.FILE+trainingFile

        dataSet = pd.read_csv(FILE_loc)

        self.data = dataSet

        self.idlist = list(dataSet['id'])

        header = list(dataSet)

        header.pop(0)

        feature_header = list()


        lastIndex = len(header) - 1

        for i in header:

            if i is header[lastIndex]:

                label = np.array(dataSet[i])

            else:

                dataSet[i] = dataSet[i].replace('?', np.nan)
                dataSet[i] = dataSet[i].fillna(0)
                vars()[i] = np.array(dataSet[i])
                vars()[i] = vars()[i].reshape(-1, 1)

                feature_header.append(i)
                self.featureDic.update({i: vars()[i]})

        self.featuresNames = feature_header
        features = np.array([])


        for i in feature_header:

            if i is feature_header[0]:

                features = vars()[i]

            else:

                features = np.concatenate((features, vars()[i]), axis=1)

        self.featureName = feature_header


        featuresTrain, featuresTest, labelTrain, labelTest = train_test_split(features, label, test_size=0.2)

        min_max_scaler_train = minmax_scale(featuresTrain)
        scaled_features_train = maxabs_scale(min_max_scaler_train)

        min_max_scaler_test = minmax_scale(featuresTest)
        scaled_features_test = maxabs_scale(min_max_scaler_test)

        startF = time.time()
        self.parkinsonForestClassifier.fit(scaled_features_train, labelTrain)
        endF = time.time()
        scoreF = self.parkinsonForestClassifier.score(scaled_features_test, labelTest)
        #print(scoreF)
        timeVarF = endF - startF
        self.timeCount.update({self.parkinsonForestClassifier: timeVarF})
        self.scoreCount.update({self.parkinsonForestClassifier: scoreF})



        startL = time.time()
        self.ParkinsonLinearClassifier.fit(scaled_features_train, labelTrain)
        endL = time.time()
        scoreL = self.ParkinsonLinearClassifier.score(scaled_features_test, labelTest)
        #print(scoreL)
        timeVarL = endL - startL
        self.timeCount.update({self.ParkinsonLinearClassifier: timeVarL})
        self.scoreCount.update({self.ParkinsonLinearClassifier: scoreL})



        startG = time.time()
        self.ParkinsonGbClassifier.fit(featuresTrain, labelTrain)
        endG = time.time()
        scoreG = self.ParkinsonGbClassifier.score(featuresTest, labelTest)
        #print(scoreG)
        timeVarG = endG - startG
        self.timeCount.update({self.ParkinsonGbClassifier: timeVarG})
        self.scoreCount.update({self.ParkinsonGbClassifier: scoreG})


        startE = time.time()
        self.ParkinsonExtraTreeClassifier.fit(scaled_features_train, labelTrain)
        endE = time.time()
        scoreE = self.ParkinsonExtraTreeClassifier.score(scaled_features_test, labelTest)
        #print(scoreE)
        timeVarE = endE - startE
        self.timeCount.update({self.ParkinsonExtraTreeClassifier: timeVarE})
        self.scoreCount.update({self.ParkinsonExtraTreeClassifier: scoreE})

        if(chButtonLogic == 1):
            self.upload_file(self.FILE, 'ecg')
            self.statusVar.set("Trained")

        else:
            self.statusVar.set("Trained")


        '''#delete this portion

        f = open("guru.txt", "w+")
        scount = "this is score count = "+str(self.scoreCount)+"\n"
        tcount = "this is time count = "+str(self.timeCount)+"\n"

        f.write(scount)
        f.write(tcount)

        f.close()'''

        #print(self.scoreCount)

        self.dataFrameSet()






    def predict(self):


        sourceFile = self.predictionFile_entry.get()
        FILE_loc = self.FILE+sourceFile

        dataSet = pd.read_csv(FILE_loc)

        dataSetNumpy = np.array(dataSet)
        dataSetNumpy = np.delete(dataSetNumpy, 0)
        features = np.delete(dataSetNumpy, -1)
        features = features.reshape(1,-1)

        result = max(self.scoreCount, key=self.scoreCount.get).predict(features)
        classifierName = self.classifiers.get(max(self.scoreCount, key=self.scoreCount.get))

        result = str(result).lstrip('[').rstrip(']')
        self.diagnosis.set(result+" (as per data label)")
        self.algoName.set(classifierName)






root = Tk()
root.geometry('600x500')
c = guiML(root)

root.mainloop()



