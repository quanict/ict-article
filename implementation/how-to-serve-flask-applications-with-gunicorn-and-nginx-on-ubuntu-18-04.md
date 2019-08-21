# How To Serve Flask Applications with Gunicorn and Nginx on Ubuntu 18.04

## Introduction
In this guide, you will build a Python application using the Flask microframework on Ubuntu 18.04. The bulk of this article will be about how to set up the [Gunicorn application server][01] and how to launch the application and configure [Nginx][02] to act as a front-end reverse proxy.

## Prerequisites
Before starting this guide, you should have:

- A server with Ubuntu 18.04 installed and a non-root user with sudo privileges. Follow our [initial server setup guide][03] for guidance.
- Nginx installed, following Steps 1 and 2 of [How To Install Nginx on Ubuntu 18.04][04].
- A domain name configured to point to your server. You can purchase one on [Namecheap][05] or get one for free on [Freenom][06]. You can learn how to point domains to DigitalOcean by following the relevant [documentation on domains and DNS][07]. Be sure to create the following DNS records:
    - An A record with `your_domain` pointing to your server's public IP address.
    - An A record with `www.your_domain` pointing to your server's public IP address.
- Familiarity with the WSGI specification, which the Gunicorn server will use to communicate with your Flask application. [This discussion][08] covers WSGI in more detail.

## Step 1 — Installing the Components from the Ubuntu Repositories

Our first step will be to install all of the pieces we need from the Ubuntu repositories. This includes pip, the Python package manager, which will manage our Python components. We will also get the Python development files necessary to build some of the Gunicorn components.

First, let's update the local package index and install the packages that will allow us to build our Python environment. These will include python3-pip, along with a few more packages and development tools necessary for a robust programming environment:
```
sudo apt update
sudo apt install python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools
```
With these packages in place, let's move on to creating a virtual environment for our project.

## Step 2 — Creating a Python Virtual Environment
Next, we'll set up a virtual environment in order to isolate our Flask application from the other Python files on the system.

Start by installing the python3-venv package, which will install the venv module:
```
sudo apt install python3-venv
```
Next, let's make a parent directory for our Flask project. Move into the directory after you create it:
```
mkdir ~/myproject
cd ~/myproject
```
Create a virtual environment to store your Flask project's Python requirements by typing:
```
python3.6 -m venv myprojectenv
```
This will install a local copy of Python and pip into a directory called `myprojectenv` within your project directory.

Before installing applications within the virtual environment, you need to activate it. Do so by typing:
```
source myprojectenv/bin/activate
```
Your prompt will change to indicate that you are now operating within the virtual environment. It will look something like this: `(myprojectenv)user@host:~/myproject$`.

## Step 3 — Setting Up a Flask Application

Now that you are in your virtual environment, you can install Flask and Gunicorn and get started on designing your application.

First, let's install `wheel` with the local instance of `pip` to ensure that our packages will install even if they are missing wheel archives:
```
pip install wheel
```
> **Note**
> Regardless of which version of Python you are using, when the virtual environment is activated, you should use the `pip` command (not `pip3`).

Next, let's install Flask and Gunicorn:
```
pip install gunicorn flask
```

### Creating a Sample App

Now that you have Flask available, you can create a simple application. Flask is a microframework. It does not include many of the tools that more full-featured frameworks might, and exists mainly as a module that you can import into your projects to assist you in initializing a web application.

While your application might be more complex, we'll create our Flask app in a single file, called `myproject.py`:
```
(myproject) $ nano ~/myproject/myproject.py
```

The application code will live in this file. It will import Flask and instantiate a Flask object. You can use this to define the functions that should be run when a specific route is requested:

~/myproject/myproject.py
```py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

if __name__ == "__main__":
    app.run(host='0.0.0.0')
```

This basically defines what content to present when the root domain is accessed. Save and close the file when you're finished.

If you followed the initial server setup guide, you should have a UFW firewall enabled. To test the application, you need to allow access to port `5000`:

```
(myproject) $ sudo ufw allow 5000
```

Now you can test your Flask app by typing:
```
(myproject) $ python myproject.py
```

You will see output like the following, including a helpful warning reminding you not to use this server setup in production:

Output
```
* Serving Flask app "myproject" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
 ```

Visit your server's IP address followed by :5000 in your web browser:
```
http://your_server_ip:5000
```

You should see something like this:

![img-01]

[img-01]: img/test_app.png (Flask sample app)


When you are finished, hit **`CTRL-C`** in your terminal window to stop the Flask development server.

### Creating the WSGI Entry Point

Next, let's create a file that will serve as the entry point for our application. This will tell our Gunicorn server how to interact with the application.

Let's call the file `wsgi.py`:
```
(myproject) $ nano ~/myproject/wsgi.py
```

In this file, let's import the Flask instance from our application and then run it:

~/myproject/wsgi.py
```py
from myproject import app

if __name__ == "__main__":
    app.run()
```

Save and close the file when you are finished.

## Step 4 — Configuring Gunicorn

Your application is now written with an entry point established. We can now move on to configuring Gunicorn.

Before moving on, we should check that Gunicorn can serve the application correctly.

We can do this by simply passing it the name of our entry point. This is constructed as the name of the module (minus the .py extension), plus the name of the callable within the application. In our case, this is wsgi:app.

We'll also specify the interface and port to bind to so that the application will be started on a publicly available interface:

cd ~/myproject
gunicorn --bind 0.0.0.0:5000 wsgi:app
You should see output like the following:

Output
[2018-07-13 19:35:13 +0000] [28217] [INFO] Starting gunicorn 19.9.0
[2018-07-13 19:35:13 +0000] [28217] [INFO] Listening at: http://0.0.0.0:5000 (28217)
[2018-07-13 19:35:13 +0000] [28217] [INFO] Using worker: sync
[2018-07-13 19:35:13 +0000] [28220] [INFO] Booting worker with pid: 28220
Visit your server's IP address with :5000 appended to the end in your web browser again:

http://your_server_ip:5000
You should see your application's output:

Flask sample app

When you have confirmed that it's functioning properly, press CTRL-C in your terminal window.

We're now done with our virtual environment, so we can deactivate it:

deactivate
Any Python commands will now use the system's Python environment again.

Next, let's create the systemd service unit file. Creating a systemd unit file will allow Ubuntu's init system to automatically start Gunicorn and serve the Flask application whenever the server boots.

Create a unit file ending in .service within the /etc/systemd/system directory to begin:

sudo nano /etc/systemd/system/myproject.service
Inside, we'll start with the [Unit] section, which is used to specify metadata and dependencies. Let's put a description of our service here and tell the init system to only start this after the networking target has been reached:

/etc/systemd/system/myproject.service
[Unit]
Description=Gunicorn instance to serve myproject
After=network.target
Next, let's open up the [Service] section. This will specify the user and group that we want the process to run under. Let's give our regular user account ownership of the process since it owns all of the relevant files. Let's also give group ownership to the www-data group so that Nginx can communicate easily with the Gunicorn processes. Remember to replace the username here with your username:

/etc/systemd/system/myproject.service
[Unit]
Description=Gunicorn instance to serve myproject
After=network.target

[Service]
User=sammy
Group=www-data
Next, let's map out the working directory and set the PATH environmental variable so that the init system knows that the executables for the process are located within our virtual environment. Let's also specify the command to start the service. This command will do the following:

Start 3 worker processes (though you should adjust this as necessary)
Create and bind to a Unix socket file, myproject.sock, within our project directory. We'll set an umask value of 007 so that the socket file is created giving access to the owner and group, while restricting other access
Specify the WSGI entry point file name, along with the Python callable within that file (wsgi:app)
Systemd requires that we give the full path to the Gunicorn executable, which is installed within our virtual environment.

Remember to replace the username and project paths with your own information:

/etc/systemd/system/myproject.service
[Unit]
Description=Gunicorn instance to serve myproject
After=network.target

[Service]
User=sammy
Group=www-data
WorkingDirectory=/home/sammy/myproject
Environment="PATH=/home/sammy/myproject/myprojectenv/bin"
ExecStart=/home/sammy/myproject/myprojectenv/bin/gunicorn --workers 3 --bind unix:myproject.sock -m 007 wsgi:app
Finally, let's add an [Install] section. This will tell systemd what to link this service to if we enable it to start at boot. We want this service to start when the regular multi-user system is up and running:

/etc/systemd/system/myproject.service
[Unit]
Description=Gunicorn instance to serve myproject
After=network.target

[Service]
User=sammy
Group=www-data
WorkingDirectory=/home/sammy/myproject
Environment="PATH=/home/sammy/myproject/myprojectenv/bin"
ExecStart=/home/sammy/myproject/myprojectenv/bin/gunicorn --workers 3 --bind unix:myproject.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
With that, our systemd service file is complete. Save and close it now.

We can now start the Gunicorn service we created and enable it so that it starts at boot:

sudo systemctl start myproject
sudo systemctl enable myproject
Let's check the status:

sudo systemctl status myproject
You should see output like this:

Output
● myproject.service - Gunicorn instance to serve myproject
   Loaded: loaded (/etc/systemd/system/myproject.service; enabled; vendor preset: enabled)
   Active: active (running) since Fri 2018-07-13 14:28:39 UTC; 46s ago
 Main PID: 28232 (gunicorn)
    Tasks: 4 (limit: 1153)
   CGroup: /system.slice/myproject.service
           ├─28232 /home/sammy/myproject/myprojectenv/bin/python3.6 /home/sammy/myproject/myprojectenv/bin/gunicorn --workers 3 --bind unix:myproject.sock -m 007
           ├─28250 /home/sammy/myproject/myprojectenv/bin/python3.6 /home/sammy/myproject/myprojectenv/bin/gunicorn --workers 3 --bind unix:myproject.sock -m 007
           ├─28251 /home/sammy/myproject/myprojectenv/bin/python3.6 /home/sammy/myproject/myprojectenv/bin/gunicorn --workers 3 --bind unix:myproject.sock -m 007
           └─28252 /home/sammy/myproject/myprojectenv/bin/python3.6 /home/sammy/myproject/myprojectenv/bin/gunicorn --workers 3 --bind unix:myproject.sock -m 007
If you see any errors, be sure to resolve them before continuing with the tutorial.

## Step 5 — Configuring Nginx to Proxy Requests

----

[08]: https://www.digitalocean.com/community/tutorials/how-to-set-up-uwsgi-and-nginx-to-serve-python-apps-on-ubuntu-14-04#definitions-and-concepts
[07]: https://www.digitalocean.com/docs/networking/dns/
[06]: http://www.freenom.com/en/index.html
[05]: https://namecheap.com/
[04]: https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04
[03]: https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-18-04
[02]: https://www.nginx.com/
[01]: http://gunicorn.org/
[source]: https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04#step-1-—-installing-the-components-from-the-ubuntu-repositories