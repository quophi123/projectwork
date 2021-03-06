from multiprocessing.dummy import dict
import base64
import streamlit as st
import pandas as pd
import numpy as np
import time
timestr = time.strftime("%Y%m%d-%H%m%S")
import fnmatch
import os
import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import zipfile
import tempfile

class LoadData: 
	@st.cache(suppress_st_warning=True)
	def load_data(self,df):
		
		if df is not None:
		
			df = pd.read_csv(df)
			st.success("Data Loaded Successfully")
		else:	
			st.error("Please Upload Data")
			st.stop()
		return df


class EditColumns:
	#replacing all null values in data for accurate results
	def replace_null(self,data):
		choice = st.sidebar.radio("Manipulate Null Values",("","Yes","No"))
		if data.isnull():
			select = st.radio("Replace or Drop",("","Replace","Drop"))
			if select == "replace":
				choice = st.sidebar.radio("Replace",("","With mean","Custom replace"))
				if choice == "With mean":
					data = data.fillna(data.mean())
				elif choice == "Custom replace":
					try:
						enter = float(st.text_input("enter replacementvalue"))
						data = data.fillna(enter)
					except ValueError as e:
						st.warning("enter numericals only")
				else:
					pass
			elif select == "Drop":
				axis=st.slider("Select axis of dropping",0,1,help="**Note** setting axis to **1** will drop all columns with null values and setting axis to **0** will drop all rows with null values")
				#thresh=st.sidebar.slider("thresh",0,5,help="**Note** setting thresh to a particular value means rows with the number or more null values will be droped **eg** thresh = **2** means rows with **2** or more null values should be droped")
				if st.button("Dropna"):
					data=data.dropna(axis=axis)
					data1 =data
					#st.dataframe(data1)
			else:
				pass
		else:
			data = data
			st.info("No null Values")
		return data



def show_sammury(data):
	return data.describe().T
#creating object for  column edit
edit_null = EditColumns()

def null_values(data):
	#data = edit_null.replace_null(data)
	pass
	

class Model:
	def add_parameter(self,clf_name):
		params = {}
		if clf_name == "KNN":
			with st.form("form_knn"):
				st.subheader(f"select your `{clf_name}` parameters here")
				col1,col2,col3 = st.beta_columns([2,1,2])
				with col1:
					n_jobs = st.slider("n_jobs",2,5)
					leaf_size = st.slider("leaf_size",10,100,step=10)
					n_neighbors = st.slider("n_neighbors",1,5,value=5)
				with col2:
					pass
				with col3:
					weights = st.radio("weights",("uniform","distance"))
					algorithm = st.radio("algorithm",("scale","auto"))
					metric_params = st.radio("metric_params",("True","Fasle"))
					metric = st.radio("metric",("minkowski","Fasle"))
				submit = st.form_submit_button("Predict")
				if submit:
					params["n_neighbors"] = n_neighbors
					params ["leaf_size"] = leaf_size
					params ["n_jobs"] = n_jobs
					params["weights"] = weights
					st.success("Predicted successfully")
		elif clf_name == "SVM":
			with st.form("form_svm"):
				st.subheader(f"select your `{clf_name}` parameters here")
				col1,col2,col3 = st.beta_columns([2,1,2])
				with col1:
					kernel_params = st.radio("kernel_params",("rbf","linear","poly","sigmoid","precomputed"))
					gamma_params = st.radio("Gamma",("scale","auto"))
				with col2:
					pass
				with col3:
					shrinking = st.radio("shrinking",(True,False))
					verbose = st.radio("verbose",(True,False))
					C = st.slider("C",1,10,value=1)
				submit = st.form_submit_button("Predict")
				if submit:
					params["C"] = C
					params["kernel_params"] = kernel_params
					params["Gamma"] = gamma_params
					params["shrinking"] = shrinking
					params["verbose"] = verbose
					st.success("Predicted successfully")
     
     
     
		elif clf_name == 'LogisticRegression':
			with st.form("form_random"):
				st.subheader(f"select your `{clf_name}` parameters here")
				col1,col2,col3 = st.beta_columns([2,1,2])
				with col1:
					class_weight = st.radio("class_weight",("None","linear"))
					C = st.slider("C",1,10)
					n_jobs = st.slider("n_jobs",1,5)
				with col2:
					pass
				with col3:
					dual = st.radio("dual",("True","False"))
					fit_intercept = st.radio("fit_intercept",("True","Fasle"))
				submit = st.form_submit_button("Build Model")
				if submit:
					params["C"] = C
					params["class_weight"] = class_weight
					params["dual"] = dual
					params["fit_intercept"] = fit_intercept
	
					params["n_jobs"] = n_jobs
					st.success("Predicted successfully")
		return params    
		
	#params = add_parameter(clf_name)
	def classifier(self,clf_name,params):
		if clf_name == 'KNN':
			classifier = KNeighborsClassifier(n_neighbors=params["n_neighbors"],leaf_size=params["leaf_size"],n_jobs=params["n_jobs"],
										weights=params["weights"])
		elif clf_name == 'SVM':
			classifier = SVC(C=params["C"],kernel=params["kernel_params"],gamma=params["Gamma"],verbose=params["verbose"],
							shrinking=params["shrinking"])
		elif clf_name == 'LogisticRegression':
			classifier = LogisticRegression()
		return classifier
			
			
	def linear_regression(self,clf_name):
		with st.form("form_random"):
			col1,col2,col3 = st.beta_columns([2,1,2])
			params={}
			with col1:
				C = st.slider("C",1,10)
				n_jobs = st.slider("n_jobs",1,5)
			with col2:
				pass
			with col3:
				fit_intercept = st.radio("fit_intercept",("True","Fasle"))
			submit = st.form_submit_button("Build Model")
			if submit:
				params["C"] = C
				params["fit_intercept"] = fit_intercept
				params["n_jobs"] = n_jobs
				st.success("Predicted successfully")
		return params
	



	def regression_classifier(self,params):
		classifire = LinearRegression(n_jobs=params["n_jobs"],fit_intercept=params["fit_intercept"])
		return classifire
			

		
		
class CleanData:    
	#function to clean data
	def null(self,data):
		choice = st.sidebar.radio("Will you like replace or drop null values",("Yes","No"))
		if choice == "Yes":
			select = st.radio("select an option to perform",("replace","drop"))
			if select == "replace":
				choice = st.radio("select type of replscement",("with mean","custom replace"))
				if choice == "with mean":
					data = data.fillna(data.mean())
				elif choice == "custom replace":
					try:
						enter = float(st.text_input("enter replacementvalue"))
						data = data.fillna(enter)
					except ValueError as e:
						st.warning("enter numericals only")
			elif select == "drop":
				axis=st.slider("axis",0,1)
				thresh=st.slider("thresh",0,5)
				if st.button("dropna"):
					data=data.dropna(axis = axis,thresh=thresh)
					st.dataframe(data)
		elif choice == "No":
			st.info("Ok")
			data = data
		return data
		


class SaveModel:

	#function to save the model
	def save_model(self,clf,save_as):
		try:
			pickle_out = open(f"{save_as}.pkl",'wb')
			pickle.dump(clf,pickle_out)
			pickle_out.close()
			st.success("Model saved sucessfully")
		except Exception as e:
			st.warning("Sorry model not saved something went wrong {}".format(e))
	
	
	
	
	# function to load the saved model 
	def load_model(self):
		try:
			pickle_in = pickle.load(open("test1.pkl",'rb'))
			st.success("Model saved sucessfully")
		except Exception as e:
			st.warning("Sorry model not saved something went wrong {}".format(e))
		return pickle_in
	




model_create = Model()

# function to create the model 
def classification(clf_name,X_train,y_train,X_test,y_test,accuracy_score):
  
	params = model_create.add_parameter(clf_name)
	clf = model_create.classifier(clf_name,params)
	clf=clf.fit(X_train,y_train)
	y_predict = clf.predict(X_test)
	acc = accuracy_score(y_test,y_predict)
	st.write(f"classifier : {clf_name}")
	st.write(f"accuracy : {acc}")
	return clf






class FeatureEngineering:
    
    
	#function to perform feature engineering of data
	def feature_engineering(self,data):
		encode_type = ["",'Label Encoding','One Hot Encoding']
		choice = st.sidebar.selectbox('Select Your Prefered Encoding type',encode_type)
		st.write(f"Your are using {choice} technique")
		category=st.multiselect("Select Columns to be Encoded",data.select_dtypes(include='object').columns)
		encode = LabelEncoder()
		data[category] = data[category].apply(encode.fit_transform)
		return data 


class Filedownload:
    
    def __init__(self):
        pass
    
    
    def download_file(data):
        timestr = time.strftime("%Y%m%d")
        csvfile = data.to_csv()
        b64 = base64.b64encode(csvfile.encode()).decode()
        new_filename = "new_file_{}.csv".format(timestr)
        st.markdown("### Download File Here")
        href = f'<a href ="data:file/csv;base64,{b64}" download="{new_filename}">Download Here</a>'
        st.markdown(href,unsafe_allow_html=True)
    
    
    
    
    def downloadmodel(model):
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        new_filename = "new_file_{}.pkl".format(timestr)
        st.markdown("### Download File Here")
        href = f'<a href ="data:file/output_model;base64,{b64}" download="{new_filename}">Download Here</a>'
        st.markdown(href,unsafe_allow_html=True)
		


class LoadModel:
    
    def __int__(self):
        pass
    
    
    def load_model(model):
        # file = io.open(model)
        # loaded_model = pickle.load(file)
        if model is not None:
            # myzipfile = zipfile.ZipFile(model)
            # with tempfile.TemporaryDirectory() as temp_dir:
            #     myzipfile.extractall(tem_dir)
            #     root_folder = myzipefile.namelist()[0]
            #     model_dir = os.path.join(temp_dir,root_folder)
            loaded = pd.read_pickle(f'{model}')
        return model_dir