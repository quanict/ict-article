# How To Deploy a Flask Application on an Ubuntu VPS

## What the `Red` Means
The lines that the user needs to enter or customize will be in red in this tutorial! The rest should mostly be copy-and-pastable.

### Introduction
Flask is a micro-framework written in Python and based on the Werkzeug and Jinja2 template engine for developing web applications. It is intended for developing web apps quickly.

## Setup
You need to have Apache already installed and running on your VPS. If this is not the case, follow Step One of our article on installing a [LAMP stack on Ubuntu](01).

## Step One— Install and Enable mod_wsgi
WSGI (Web Server Gateway Interface) is an interface between web servers and web apps for python. Mod_wsgi is an Apache HTTP server mod that enables Apache to serve Flask applications.

Open terminal and type the following command to install mod_wsgi:

```
sudo apt-get install libapache2-mod-wsgi python-dev
```
To enable mod_wsgi, run the following command:
```
sudo a2enmod wsgi 
```
## Step Two – Creating a Flask App

In this step, we will create a flask app. We will place our app in the `/var/www` directory.

Use the following command to move to the `/var/www` directory:
```
cd /var/www 
```

Create the application directory structure using mkdir as shown. Replace "`FlaskApp`" with the name you would like to give your application. Create the initial directory `FlaskApp` by giving following command:

```
sudo mkdir FlaskApp
```

Move inside this directory using the following command:

```
cd FlaskApp
```

Create another directory FlaskApp by giving following command:

```
sudo mkdir FlaskApp
```


Then, move inside this directory and create two subdirectories named static and templates using the following commands:
```
cd FlaskApp
sudo mkdir static templates
```
Your directory structure should now look like this:
```
|----FlaskApp
|---------FlaskApp
|--------------static
|--------------templates
```
Now, create the __init__.py file that will contain the flask application logic.
```
sudo nano __init__.py 
```
Add following logic to the file:
```python
from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello():
    return "Hello, I love Digital Ocean!"
if __name__ == "__main__":
    app.run()
```
Save and close the file.

## Step Three – Install Flask

Setting up a `virtual environment` will keep the application and its dependencies isolated from the main system. Changes to it will not affect the cloud server's system configurations.

In this step, we will create a virtual environment for our flask application.

We will use pip to install virtualenv and Flask. If pip is not installed, install it on Ubuntu through apt-get.
```
sudo apt-get install python-pip 
```

If virtualenv is not installed, use pip to install it using following command:

```
sudo pip install virtualenv 
```

Give the following command (where venv is the name you would like to give your temporary environment):

```
sudo virtualenv venv
```

Now, install Flask in that environment by activating the virtual environment with the following command:
```
source venv/bin/activate 
```

Give this command to install Flask inside:
```
sudo pip install Flask 
```

Next, run the following command to test if the installation is successful and the app is running:
```
sudo python __init__.py 
```
It should display “Running on http://localhost:5000/” or "Running on http://127.0.0.1:5000/". If you see this message, you have successfully configured the app.

To deactivate the environment, give the following command:
```
deactivate
```

## Step Four – Configure and Enable a New Virtual Host
Issue the following command in your terminal:
```
sudo nano /etc/apache2/sites-available/FlaskApp
```
> NOTE: Newer versions of Ubuntu (13.10+) require a ".conf" extension for VirtualHost files -- run the following command instead:

```
sudo nano /etc/apache2/sites-available/FlaskApp.conf
```

Add the following lines of code to the file to configure the virtual host. Be sure to change the ServerName to your domain or cloud server's IP address:
```
<VirtualHost *:80>
		ServerName mywebsite.com
		ServerAdmin admin@mywebsite.com
		WSGIScriptAlias / /var/www/FlaskApp/flaskapp.wsgi
		<Directory /var/www/FlaskApp/FlaskApp/>
			Order allow,deny
			Allow from all
		</Directory>
		Alias /static /var/www/FlaskApp/FlaskApp/static
		<Directory /var/www/FlaskApp/FlaskApp/static/>
			Order allow,deny
			Allow from all
		</Directory>
		ErrorLog ${APACHE_LOG_DIR}/error.log
		LogLevel warn
		CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```
Save and close the file.

Enable the virtual host with the following command:
```
sudo a2ensite FlaskApp
```
## Step Five – Create the .wsgi File

Apache uses the .wsgi file to serve the Flask app. Move to the /var/www/FlaskApp directory and create a file named flaskapp.wsgi with following commands:
```
cd /var/www/FlaskApp
sudo nano flaskapp.wsgi 
```
Add the following lines of code to the flaskapp.wsgi file:
```py
#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/FlaskApp/")

from FlaskApp import app as application
application.secret_key = 'Add your secret key'
```
Now your directory structure should look like this:
```
|--------FlaskApp
|----------------FlaskApp
|-----------------------static
|-----------------------templates
|-----------------------venv
|-----------------------__init__.py
|----------------flaskapp.wsgi
```
## Step Six – Restart Apache
Restart Apache with the following command to apply the changes:
```
sudo service apache2 restart 
```
You may see a message similar to the following:
```
Could not reliably determine the VPS's fully qualified domain name, using 127.0.0.1 for ServerName 
```
This message is just a warning, and you will be able to access your virtual host without any further issues. To view your application, open your browser and navigate to the domain name or IP address that you entered in your virtual host configuration.

You have successfully deployed a flask application.

Article Submitted by: Kundan Singh

---

[01]: https://www.digitalocean.com/community/articles/how-to-install-linux-apache-mysql-php-lamp-stack-on-ubuntu
[source]: https://www.digitalocean.com/community/tutorials/how-to-deploy-a-flask-application-on-an-ubuntu-vps
