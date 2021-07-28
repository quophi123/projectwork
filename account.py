import streamlit as st
from database import *
import hashlib
import re
import pandas as pd
from datetime import *



def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'




class Account:
	def login():
		st.sidebar.subheader('Enter Your Details To Login')
		username = st.sidebar.text_input('Username')
		password = st.sidebar.text_input('Password',type='password')
		if st.sidebar.checkbox('Login'):
			hashsed_pswd = generate_hashes(password)
			results = login_user(username,verify_password(password,hashsed_pswd))
			if results:
				st.success('Login Successful, Welcome {}'.format(username))
			else:
				st.warning("Please Enter Valid Credentials ")




	def sign_up():
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


