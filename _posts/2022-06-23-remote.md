---
tags：linux
---
# 关于vscode远程链接 （remote about vscode）
> due to some problem ,i can not connet my server in vscode ,but it worked in command .
> 发现终端连接可以，但是vscode不行

仔细发现有错误
**Could not chdir to home directory /home/dp: No such file or directory /usr/bin/xauth: error in locking authority file /home/dp/.Xauthority**

## 解决方法

```
sudo mkdir /home/dp
sudo usermod --shell /bin/bash --home /home/dp dp
sudo chown -R dp:dp /home/dp
```

- 解决Could not chdir to home directory /home/dp: No such file or directory /usr/bin/

问题转化为
**/usr/bin/xauth:  file /home/dp/.Xauthority does not exist**


- Step 1: Login with the required user and go to home directory.

- Step 2: Rename and backup the existing .Xauthority file.
```
mv .Xauthority old.Xauthority
``` 
- Step 3: Touch otherwise xauth with complain unless ~/.Xauthority exists
```
touch ~/.Xauthority
```
- Step 4: Only this one key is needed for X11 over SSH

```xauth generate :0 . trusted```

- Step 5: Generate our own key, xauth requires 128 bit hex encoding

```xauth add ${HOST}:0 . $(xxd -l 16 -p /dev/urandom)```
- Step 6: To view a listing of the .Xauthority file

到这里基本就解决了