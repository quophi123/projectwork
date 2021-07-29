import streamlit as st
import pandas as pd
#import functions as fnx  
from functions import Model,SaveModel,EditColumns,LoadData,FeatureEngineering
from account import *
import numpy as np
import io
import datetime
import time
import seaborn as sns
sns.set_palette('Set2')
import matplotlib
import matplotlib.pyplot as plt
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
plt.style.use('seaborn-deep')
import json
import requests
import codecs
import matplotlib.pyplot as plt
import sweetviz as sv
import streamlit.components.v1 as components
from database import *
import pickle
import os
from functions import *


def display(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width, height=height,scrolling=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

#creating object for model saving and uploading
saveModel = SaveModel()
account = Account()
feature = FeatureEngineering()





#st.title('Welcome To DS Studio')
title_template="""
<div style="background-color:#025246; padding:10px">
<h2 style="color:white;text-align:center;">Welcome To DS Studio</h2>
</div>
"""
st.markdown(title_template,unsafe_allow_html=True)



def main():
    #lets create object for class loadData
    menu = ['Home','Login','Create Account','About']
    choices = st.sidebar.selectbox("Menu",menu)


    if choices == 'Home':
        
        st.subheader('Home')
        st.subheader('Welcom To The HomePage')
        # if data is not None:
        #     col1 = st.multiselect("select",data.columns)
        #     col2 = st.multiselect("select1",data.columns)
        #     report = sv.compare(data[col1],data[col2])
        #     report.show_html(open_browser=False)
        #     display("SWEETVIZ_REPORT.html")
    
    elif choices == 'Login':
        col = st.beta_columns(1)
        st.sidebar.subheader('Enter Your Details To Login')
      
        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password',type='password')
        login = st.sidebar.checkbox('login')
        if login:
            hashsed_pswd = generate_hashes(password)
            results = login_user(username,verify_password(password,hashsed_pswd))
            if results:
                st.success('Login Successful, Welcome {}'.format(username))
                load = LoadData()
                df = st.file_uploader("Upload Your Dataset Here",('csv','xlsx'))
                new_df = load.load_data(df)
                data = pd.DataFrame(new_df)
                st.subheader('This is Activity Page')
                activitymenu = ['EDA','Data Visualization','Build Model','Make Prediction']
                activity = st.sidebar.selectbox("Select an activity",activitymenu)
                
                if activity == 'EDA':
                    st.subheader("Explore your data and analyze them here with sample built in")

                    st.sidebar.write("selection of")
                    #sidebar for EDA activities
                    eda_activities = ["select","View Data","Show Shape","Show Columns","Show Summary","Value Count","View Null Values","View Data Types"]
                    choices = st.radio("Select an EDA Activity",eda_activities,help="explore your `data` here")





                    #selecting an eda activity
                    if choices == "View Data":
                        st.subheader("Viewing **`Data`**")
                        display_mode = ["select mode","By Rows","By Columns"]
                        mode = st.selectbox("Select Mode Of View",display_mode,help="Selects either to view `Row wise` or `Column wise`")


                        #displaying data base on columns
                        if mode == "By Rows":
                            st.subheader("viewing by ***`row`***")
                            show = st.radio("Select",("choose how","display head","display tail","prefered number"))


                            #displays first ten(10) rows of the data
                            if show == "display head":
                                st.subheader("Showing first **`ten(10)`** of the dataset")
                                st.dataframe(data.head())

                            #displaing last ten(10) rows of the data
                            elif show == "display tail":
                                st.subheader("Showing last **`ten(10)`** of the dataset")
                                st.dataframe(data.tail())


                            #displaying your prefered number of rows
                            elif show == "prefered number":
                                number = st.slider("slide to your prefered number",min_value=1,max_value=data.shape[0])
                                st.subheader("Showing `{}` rows of the dataset".format(number))
                                st.dataframe(data.head(number))


                        #displaying data base on the columns selected by user
                        elif mode == "By Columns":
                            """Displaying Data by the columns select by the user"""
                            st.subheader("Viewing by ***`column`***")
                            columns_to_display = st.multiselect("select columns to display",data.columns)
                            selected_colums = data[columns_to_display]
                            st.dataframe(selected_colums)



                    elif choices == "Show Shape":
                        st.subheader("Showing `shape` of Data")
                        dimension = ['Columns','Rows','All']
                        dimensions = st.radio('Select Dimesion',dimension)
                        if dimensions == 'Columns':
                            st.write(f'Showing {dimensions} of Dataset')
                            st.write(f'The Data has `{data.shape[1]}` Columns')
                        elif dimensions == 'Rows':
                            st.write(f'Showing {dimensions} of Dataset')
                            st.write(f'The Data has `{data.shape[0]}` Rows')
                        elif dimensions == 'All':
                            st.write('Showing Shape of Dataset')
                            st.write("The dataset has **`{}`** rows and **`{}`** columns as shape **`{}`**".format(data.shape[0],data.shape[1],data.shape))
                            
                            
                            

                    elif choices == "Show Columns":
                        st.subheader("Showing Columns of **`{}`**".format(data.shape[1]))
                        st.dataframe(data.columns)

                    elif choices == "Show Summary":
                        st.subheader("Show Summary of **`Data`**")
                        st.write(data.describe().T)

                    elif choices == "Value Count":
                        st.subheader("Counting the number of `unique` value in each column and the unique values in the various columns")
                        columns_to_display = st.selectbox("",data.columns)
                        selected_colums = data[columns_to_display].value_counts()
                        selected_colums1 = data[columns_to_display].unique()
                        st.dataframe(selected_colums)
                        st.write('Shows unique values in of the column `{}`'.format(columns_to_display))
                        st.dataframe(selected_colums1)


                    #displaying null values in the dataset
                    elif choices == "View Null Values":
                        """ **Null value means empty values not zero**"""
                        st.subheader("View Null Values ***Hint*** **`0`** means is not `null` and **`1`** means is `null`")
                        st.dataframe(data.isna())


                    #displaying the data types of the various column in the dataset
                    elif choices == "View Data Types":
                        st.subheader("Showing data types of the various columns")
                        st.write(data.dtypes)


                        
                elif activity == 'Data Visualization':
                    st.subheader('Your are looking at **`{}`**'.format(activity))
                    st.subheader('**Visualize** Your `Data` Here')
                    


                        #pie chart plotting
                    if st.sidebar.checkbox("Pie Chart"):
                        st.subheader("Pie Chart")
                        column = data.select_dtypes(['float32','float64','int32','int64']).columns
                        pie_values = st.selectbox("Select columns to plot",column,help="select your prefered plot for `visualizations`")
                        #columns_to_plot = st.multiselect('Selects the columns to plot',data.columns)
                        fig1,ax1 = plt.subplots()
                        ax1.pie(data[pie_values].value_counts(),labels=data[pie_values].value_counts())
                        ax1.axis('equal')
                        st.pyplot(plt.show())
                        #pie_plot = data[pie_values].value_counts().plot.pie(autopct="%1.1f%%")
                    # if st.button("Generate Pie plot"):
                        #    st.write( pie_plot)
                        #    st.pyplot()
                        
                        



                #Correlation plotting
                    if st.sidebar.checkbox("Correlation Matrics"):
                        st.subheader("Correlation Matrics")
                        if st.button("Generate Correlation plot"):
                            st.write(sns.heatmap(data.corr(),annot=True))
                            st.pyplot()


                #bar plotting
                    if st.sidebar.checkbox("Bar Plot"):
                        st.subheader("Bar Plot")
                        column = data.select_dtypes(['float32','float64','int32','int64']).columns
                        bar_values = st.selectbox("Select first column",column,help="select your prefered plot for `visualizations`")
                        if st.button("Generate bar plot"):
                            st.success(f"Genarating line plot for {bar_values}")
                            st.write( sns.distplot(data[bar_values],kde=False))
                            st.pyplot()
                        features = data.columns
                        sns.set_style('white')
                    
                        st.pyplot(sns.FacetGrid(data,col='{}'.format(bar_values)))
                #line plotting
                    if st.sidebar.checkbox("Line Plot"):
                        st.subheader("Line Plot")
                        column = data.select_dtypes(['float32','float64','int32','int64']).columns
                        plot_type = st.selectbox("Select either single column plot or double",("Single","Double"))
                        if plot_type == "Single":
                            if st.button("Generate Single Column line plot"):
                                line_values = st.selectbox("Select second column",column,help="select your prefered plot for `visualizations`")
                                st.success(f"Genarating line plot for {line_values}")
                                plt.plot(data[f'{line_values}'])
                                st.pyplot(plt.show())
                        elif plot_type == "Double":
                                if st.button("Generate Double Columns line plot"):
                                    line_values = st.selectbox("Select column first",column,help="select your prefered plot for `visualizations`")
                                    line_values1 = st.selectbox("Select second column",column,help="select your prefered plot for `visualizations`")
                                    st.success(f"Genarating line plot for {line_values} against {line_values1}")
                                    x=data[f'{line_values}']
                                    y=data[f'{line_values1}']
                                    plt.plot(x,y)
                                    plt.xlabel(f'{line_values}')
                                    plt.ylabel(f'{line_values1}')
                                    plt.title(f" {line_values} against {line_values1}")
                                    st.pyplot(plt.show())


                #Histogram plotting
                    if st.sidebar.checkbox("Histogram"):
                        st.subheader("Histogram")
                        column = data.select_dtypes(['float32','float64','int32','int64']).columns
                        st.success(f"Genarating Histogram plot for")
                        plt.hist(column)
                        plt.legend(loc='upper right')

                        st.pyplot(plt.show())


                #Scatter plots
                    if st.sidebar.checkbox("Scatter Plot"):
                        st.subheader("Scatter Plot")
                        column = data.select_dtypes(['float32','float64','int32','int64']).columns
                        scatter_values1 = st.selectbox("Select first column to plot",column,help="select your prefered plot for `visualizations`")
                        scatter_values2 = st.selectbox("Select second column to plot",column,help="select your prefered plot for `visualizations`")
                        #line1_values = st.selectbox("Select second column",data.columns,help="select your prefered plot for `visualizations`")
                        if st.button("Generate scatter plot"):
                            st.success(f"Genarating line plot for {scatter_values1} against {scatter_values2}")
                            sns.scatterplot(x=data[f'{scatter_values1}'],y=data[f'{scatter_values2}'])
                            st.pyplot(plt.show())
                        
                    
                    
                elif activity == 'Data Cleaning': 
                    #creating oblect for column editing and data types of colmns
                    edit_cols = EditColumns()
                    clean_data_object = CleanData()
                    st.subheader('Your are looking at **`{}`**'.format(activity))
                    clean_type = ["",'Null Values','Add Columns','Change Data Types']
                    clean = st.sidebar.radio("Choose Type of Cleaning",clean_type)
                    
                    if clean == 'Null Values':
                        data= clean_data_object.null(data)
                        st.dataframe(data)
                    elif clean == 'Add Columns':
                        new_data = edit_cols.get_newcolumn_names(data)
                        st.dataframe(new_data)
                    elif clean == 'Change Data Types':
                        data=  clean_data_object.change_column_datatype(data)
                        st.dataframe(data)
                        
                        
                        
                    # perform feature engineering of dataset       
                elif activity == 'Feature Engineering':
                    st.subheader("Feature engineering page")   
                    st.dataframe(FeatureEngineering.feature_engineering(data))
                    
                elif activity == 'Build Model':
                    st.subheader('Your are looking at **`{}`**'.format(activity))
                    st.subheader("Make Your **`Machine Learning`** Models here")

                    if st.sidebar.checkbox("View Preprocessed Data",help="This displays preprocessed `data`"):
                        st.subheader("Showing Preprocessed **`data`**")
                        with st.beta_expander("Expend to show data details"):
                            st.dataframe(data)
                    #sidebar for selecting target columns of the datasets for model building
                    
                    
                    def target(data):
                        #try:
                            #if st.sidebar.checkbox("Select Target Columns",help="This columns are uesd as referenced for `prediction`"):
                        st.subheader("Select Target  **`Columns`**") 
                        selected_target=st.selectbox("Select",data.columns,help="This columns are uesd as referenced for `prediction`")
                        #except KeyError as e:
                            #st.warning("Please check Select Features check box")
                            # st.stop()
                        return selected_target 
                        
                    #sidebar for selecting features of the datasets for model building
                    def features(data):
                        # try:
                            # if st.sidebar.checkbox("Select Features",help="This columns are uesd as the dependent variable for the `target` "):
                        st.subheader("Select Features **`data`**")
                        selected_features=st.multiselect("Select",data.columns,help="This columns are uesd as referenced for `prediction`")
                        #except KeyError as e:
                            #st.warning("Please check Select Features check box")
                            #st.stop()
                        return selected_features
                    
                    
                    target = target(data)
                    features = features(data)
                    
                
                    y = data[target]
                    X = data[features]
                    test_size=st.slider("Enter Test Size",0.1,0.2,0.1,0.05,help="This takes care of the `percentage` of your data you want to use for `testing`")
                    random_state=st.slider("Enter Random State",1,5,help="This selects the `ramdom state` of your model")
                        
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12345)
                    col1,col2 = st.beta_columns([1,4])
                    col1.subheader('Selected Target is')
                    col1.write(y)
                    col2.subheader("Selected Features are")
                    col2.write(X)
                    
                        #sidebar for selecting the type of model to build
                    model_type = ["Select prefered algorithm","Regression","Classification"]
                    model = st.sidebar.selectbox("Select Model Type",model_type,help="models are algorithms for `predicting` the `relationdhip` between data")
                    
                    
                    
                    
                    
                    #object for creating model
                    model_obj = Model()
                    
                    
                    
                #Regression model
                    if model == "Regression":
                        st.subheader("Regression Page")
                        try:
                            mod=model_obj.classification(clf_name,X_train,y_train,X_test,y_test,accuracy_score)
                            save = st.info(" Do You Save Model")
                            enter_model_name = st.text_input("Eneter how to save your model")
                            #if st.button("Save Model"):
                            saveModel.save_model(mod,save_as="test1")
                        except KeyError as e:
                            st.error("Please Submit Parameters")
                        
                        



                #classification model
                    elif model == "Classification":
                        st.subheader("Classification Page")
                        classifier_type = ["KNN","SVM","LogisticRegression"]
                        clf_name = st.sidebar.selectbox("Select {} Type".format(model),classifier_type,help="select the type of `classifier` you want to use")
                        try:
                            mod=classification(clf_name,X_train,y_train,X_test,y_test,accuracy_score)
                            save = st.info(" Do You Save Model")
                            save_as = st.text_input("Eneter how to save your model")
                            if st.checkbox('save'):
                                saveModel.save_model(mod,save_as='text1')
                        except KeyError as e:
                            st.error("Please Submit Parameters")
                            
                        
                elif activity == "Make Prediction":
                
                    st.subheader("Make Your Preditions Here")
                    try:
                        predict_values = int(st.text_input("Enter the number of columns to be entered"))
                        dic =[]
                        for i in range(predict_values):
                            predict_values = st.number_input(f"Coulumn{i+1}")
                            dic.append( predict_values)
                        st.write(dic)
                        
                        loaded_model = pickle.load(open("test1.pkl",'rb'))
                    
                        loan=loaded_model.predict(np.array(dic).reshape(1,-1))
                        st.success("The Predicted Plant is{}".format(loan))
                    except Exception as e:
                        st.warning(f"Please Enter number of columns {e}")
                            
            else:
                st.warning("Please Enter Valid Credentials ")           
                        

            #Account creation
    elif choices == "Create Account":
        st.subheader("Create A New Account Here")
        st.subheader('Fill The Form Below With Your Details To Create An Account')
        with st.form('my_form'):
            new_username = st.text_input("User name")
            new_email = st.text_input("Enter a valid email")
            new_password = st.text_input("Password",type="password")
            confirm_password = st.text_input("Confirm Password",type="password")
            date_created = datetime.now().strftime("%B %d,%Y")
            if new_password == confirm_password:
                st.success("Passwords Confirmed ")
            else:
                st.warning("Both Passwords Should be the same or not empty")
            submit = st.form_submit_button("Create Account")
        if submit:
            if new_username and new_email and new_password != " ":
                if (re.search(regex,new_email)):
                    if new_password == confirm_password:
                        create_usertable()
                        hashed_new_password = generate_hashes(new_password)
                        add_data(new_username,new_email,hashed_new_password,date_created)
                        st.success(f"Account for {new_username} created Successfully")
                        st.info("Login at the sidebar")
                    else:
                        st.warning("Please both passwords must be same")
                else:
                    st.warning("please enter a valid email")
            else:
                st.warning("No Field Should Be Empty")

    else:
        st.subheader(' {} Project '.format(choices))
        st.write("This is a Data Science Process automation web app Developed by 4606318 under the supervision of `DR Frimpong Twum` a Senior Lectural at the Department of Computer Science KNUST")
        
if __name__=="__main__":
    main()