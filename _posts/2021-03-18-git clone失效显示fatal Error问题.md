---
tags: linux应用
---
# git clone 失效显示 fatal
## 解决方案
github代理问题
**首先终端输入下面命令**
```bash
git config --global http.proxy
```
如果没有输出，则未设置Git Bash中的代理

然后使用这些命令进行设置，并使用代理和端口
```bash
 git config --global http.proxy proxyaddress:port
```
取消代理的话输入以下命令
```bash
git config --global --unset http.proxy
```