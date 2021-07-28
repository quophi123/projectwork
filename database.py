
import sqlite3

conn = sqlite3.connect("Usersdatabase.db", check_same_thread=False)
c = conn.cursor()

#functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS User(username TEXT,email TEXT,password TEXT, date DATE)')

def add_data(new_username,new_email,new_password,date_created):
	c.execute('INSERT INTO User(username,email,password,date) VALUES (?,?,?,?)',(new_username,new_email,new_password,date_created))
	conn.commit()


def view_profiles():
	c.execute('SELECT * FROM User')
	data = c.fetchall()
	return data


def login_user(username,password):
	c.execute('SELECT username,password FROM User WHERE username =? AND password=?',(username,password))
	data = c.fetchall()
	return data


