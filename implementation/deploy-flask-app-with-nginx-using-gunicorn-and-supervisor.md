# Deploy flask app with nginx using gunicorn and supervisor

![img-01]

In this blog, I will discuss how to set a Flask app up on an Ubuntu server.

This is a walkthrough that illustrates how to deploy a Flask application using an easy technique.

> We will be using the following technologies:

> [Flask](01): Server backend

> [Nginx](02): Reverse proxy

> [Gunicorn](03): Deploy flask app

> [Supervisor](04): Monitor and control gunicorn process

## Install required packages

```
sudo apt-get install nginx supervisor python-pip python-virtualenv
```

## Create a virtual environment

If you are not using python virtual environments, you should! Virtual environments create isolated python environments. This allows to run multiple versions of library on the same machine.

Let’s create a virtual environment.
```
$ virtualenv .env
```

And activate it.
```
$ source .env/bin/activate
```

## Create a Flask app

Install Flask and other dependencies.
```
$ pip install Flask
$ pip install -r requirements.txt
```
Write the code for Flask app.
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)
```

You can run the Flask app simply by running the following command:
```
$ python app.py
```

But it is not safe to use the Flask development server for a production environment. So, you can use Gunicorn to server our python code.

## Setup Gunicorn

> Gunicorn ‘Green Unicorn’ is a Python WSGI HTTP Server for UNIX.

Install gunicorn.
```
$ pip install gunicorn
```

Let’s start a Gunicorn process to serve your Flask app.
```
gunicorn app:app -b localhost:8000 &
```

To stop gunicorn
```
pkill -f gunicorn
```

You can make Gunicorn process listen to any open port.

This will set your Gunicorn process off running in the background, which will work fine for your purposes here. An improvement that can made here is to run Gunicorn via Supervisor.

### Use supervisor

> Supervisor allows to monitor and control a number of processes on UNIX-like operating systems.

Supervisor will look after the Gunicorn process and make sure that they are restarted if anything goes wrong, or to ensure the processes are started at boot time.

Create a supervisor file in `/etc/supervisor/conf.d/` and configure it according to your requirements.

```conf
[program:hello_world]
directory=/home/ubuntu/hello_world
command=/home/ubuntu/.env/bin/gunicorn app:app -b localhost:8000
autostart=true
autorestart=true
stderr_logfile=/var/log/hello_world/hello_world.err.log
stdout_logfile=/var/log/hello_world/hello_world.out.log
```

To enable the configuration, run the following commands:
```
sudo supervisorctl reread
sudo service supervisor restart
```
This should start a new process. To check the status of all monitored apps, use the following command:
```
sudo supervisorctl status
```

## Setup nginx
> Nginx is an HTTP and reverse proxy server.

Let’s define a server block for our flask app.
```
sudo vim /etc/nginx/conf.d/virtual.conf
```
Paste the following configuration:

```conf
server {
    listen       80;
    server_name  your_public_dnsname_here;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```
Proxy pass directive must be the same port on which the gunicorn process is listening.
Restart the nginx web server.
```
$ sudo nginx -t
$ sudo service nginx restart
```

Now, if you visit your public DNS name in a web browser, you should see Hello World page.
![img-02]

Congratulations! Now the Flask app is successfully deployed using a configured nginx, gunicorn and supervisor.
There are some other ways to deploy flask app which are as follows:
- Using uwsgi
- Using gevent
- Using twisted web

----

[img-02]: img/1_NcJJMY4OaVj5H6YN1ZKkVQ.png
[04]: http://supervisord.org/
[03]: http://gunicorn.org/
[02]: https://www.nginx.com/
[01]: http://flask.pocoo.org/
[img-01]: img/1_nFxyDwJ2DEH1G5PMKPMj1g.png
[source]: https://medium.com/ymedialabs-innovation/deploy-flask-app-with-nginx-using-gunicorn-and-supervisor-d7a93aa07c18