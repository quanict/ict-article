# Git

## Config

### remote url
```
git config --get remote.origin.url
```

### Changing a remote's URL

```
git remote set-url
```

List your existing remotes in order to get the name of the remote you want to change.

```
$ git remote -v
> origin  git@github.com:USERNAME/REPOSITORY.git (fetch)
> origin  git@github.com:USERNAME/REPOSITORY.git (push)
```

Change your remote's URL from SSH to HTTPS with the git remote set-url command.


```
git remote set-url origin https://github.com/USERNAME/REPOSITORY.git
```

Verify that the remote URL has changed.

```
$ git remote -v
# Verify new remote URL
> origin  https://github.com/USERNAME/REPOSITORY.git (fetch)
> origin  https://github.com/USERNAME/REPOSITORY.git (push)
```

## Remove

### Remove directory

```
git rm -r --cached [directory-name]
```

## Remember

```
eval `ssh-agent -s`
ssh-add ~/.ssh/*_rsa
```
