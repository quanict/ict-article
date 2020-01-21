# 10 Most Used Nginx Commands Every Linux User Must Know

## Install Nginx Server

```
$ sudo yum install epel-release && yum install nginx   [On CentOS/RHEL]
$ sudo dnf install nginx                               [On Debian/Ubuntu]
$ sudo apt install nginx                               [On Fedora]
```


## Check Nginx Configuration Syntax


```
$ sudo nginx -t

nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```


You can test the Nginx configuration, dump it and exit using the `-T` flag as shown.


```
$ sudo nginx -T
```

------------------
[source]: https://www.tecmint.com/useful-nginx-command-examples/