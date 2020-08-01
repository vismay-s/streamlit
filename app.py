import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score,confusion_matrix,f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import brier_score_loss

def main():
    st.title(" WebApp.MLtrain(coders,non-coders) ")
    st.sidebar.title("Clf.(parameters,metrics)")
    st.markdown("Hello, My name is Vismay. I created this interface to help non-coding background people get started with ML. The dataset is a simple Oranges vs Grapefruit set.")
    st.markdown("The aim is to simply understand impact of parameter tuning and classifier strenghts and evaluation metrics. So...")
    st.subheader(" WebApp.MLfit( build = mouse_click, learn = True, code = False)")
    st.sidebar.markdown("Is a specific fruit orange or a grapefruit?")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("C:\zdataskills\streamlit\streamlit-ml\citrus.csv")
        labelencoder=LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data
    
    @st.cache(persist=True)
    def split(df):
        y = df.name
        x = df.drop(columns=['name'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
    
        if cmap is None:
            cmap = plt.get_cmap('Blues')
    
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
    
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()    
        #my scikit learn version do not support direct importing of this library. Also due to memory issues I havent used this in my code. But it can make UI more appealing

    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            #st.subheader("Confusion Matrix")
            st.write("Confusion Matrix:",confusion_matrix(y_test,y_pred))
            st.write("Confusion Matrix: True positive (tp), True negative (tn), False positive (fp) and False negative (fn) predictions")
            #plot_confusion_matrix(cm=cm,normalize = False, target_names = ['high', 'medium'])
            
        if 'F1 score' in metrics_list:
             st.write("F1 Score: ", f1_score(y_test, y_pred, labels=class_names).round(2))
             st.write("F1 Score: Harmonic mean between precision and recall.")
             

        if 'Cohen Kappa' in metrics_list:
             st.write("Cohen Kappa :", cohen_kappa_score(y_test, y_pred).round(2))
             st.write("Cohen Kappa: How much better is your model over the random classifier that predicts based on class frequencies.")
             
        if 'Brier Loss' in metrics_list:
            st.write("Brier Loss:",brier_score_loss(y_test, y_pred).round(2))
            st.write("Brier Loss: Measure of how far your predictions lie from the true values.")
            
        
    df = load_data()
    class_names = ['orange', 'grapefruit']
    
    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("K Neighbors Classifier", "Logistic Regression","Support Vector Machine (SVM)","MLP Classifier", "Random Forest"))
    
    if classifier == 'K Neighbors Classifier':
        st.sidebar.subheader("Model Hyperparameters")
        N = st.sidebar.number_input("N_neighbors (Neighbor parameter)", 1, 20, step=1, key='N')
        weight = st.sidebar.radio("Weights", ("uniform", "distance"), key='weight')
        algorithm = st.sidebar.radio("Algorithm", ("brute", "auto"), key='algoritm')

        metrics = st.sidebar.multiselect("What metrics to evaluate?", ('Confusion Matrix', 'F1 score','Cohen Kappa','Brier Loss'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("K Neighbors Classifier Results")
            model = KNeighborsClassifier(n_neighbors=N, weights=weight, algorithm=algorithm)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            
            plot_metrics(metrics)

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to evaluate?", ('Confusion Matrix', 'F1 score','Cohen Kappa','Brier Loss'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                        
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Inverse of regularization strength)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to evaluate?", ('Confusion Matrix', 'F1 score','Cohen Kappa','Brier Loss'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            
            plot_metrics(metrics)
            
    if classifier == 'MLP Classifier':
        st.sidebar.subheader("Model Hyperparameters")
        activation = st.sidebar.radio("Activation Function", ("tanh", "relu"), key='activation')
        solver=st.sidebar.radio("Solver", ("adam", "sgd"), key='solver')
        alpha=st.sidebar.number_input("Alpha (L2 penalty)", 0.01, 1.0, step=0.01, key='alpha')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to evaluate?", ('Confusion Matrix', 'F1 score','Cohen Kappa','Brier Loss'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("MLP Classifier Results")
            model = MLPClassifier(activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
            
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to evaluate?", ('Confusion Matrix', 'F1 score','Cohen Kappa','Brier Loss'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Orange vs Grapefruit")
        st.write(df)
        st.markdown("Whether the given fruit is an orange or a grapefruit")
        
    if st.sidebar.checkbox("What is this?", False):
        st.subheader("What is this App?")
        st.markdown("I am Vismay Sudra, an electrical engineering student. I created this simple app for two reasons:")
        st.markdown(" 1: Get acquainted with streamlit functionalities. (I am a learner too)")
        st.markdown(" 2: Get python beginners and non coders acquainted with ML Classifiers."
                    " Hope you like it!") 
        
    if st.sidebar.checkbox("So am I a Data Scientist now?", False):
        st.subheader("NO,a few hundred hours to go")
        st.markdown("This was only Binary Classification. But you sure did get acquainted with various classifiers. Also antime in doubt, you can simply use the UI to get an estimate evaluation")    
        
        

if __name__ == '__main__':
    main()


