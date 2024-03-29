---
tags: openEuler
---
# FAQ
## 安装检测工具
```bash
yum install lsof -y
```
## ssh密钥连接失败
应该是 ssh 免密钥认证到 14.0.0.13 失败
如图所示 

![](https://i.loli.net/2021/07/06/dlT4KruHESVR7eN.png)

**解决方法：**
应为没有设置免密公钥通信所以建议直接密码接入
```
tiup cluster deploy tidb-x v5.1.0 --user root -p
```
成功
## tiup playground集群启动失败

![](https://i.loli.net/2021/07/08/2osJhVgZxT6ScDy.png)

4000端口
**解决方法**
忽略改为单机cluster部署

## 单机部署失败

![](https://i.loli.net/2021/07/09/eriIbkXT4sLZBAo.png)

因为之前一直部署不上，查看具体文件
发现该文件下为X86无法执行
本机为arm架构

**解决方法：**
在部署的yaml文件中的global下加入`arch: "arm64"`
## 部署成功无法启动tiflash

**解决方法：**
在部署的yaml文件中注释掉tiflash配置
>因为tidb的主要配件就三个 `PD` `tidb` `tikv` 所以其他组件可以暂时不用
改完后卸载重装
启动成功

## 上述过程中如果遇到某组件启动失败都可以lsof -i:端口号查看启动情况然后kill -9 杀死该进程，再start试一试

每个文件都会记录报错日志，可以下载日志寻找方法

## 测试把磁盘占满了，tikv全部挂掉
这种状况装什么工具都是失败，基本所有命令都失败，因为没有内存空间了
如图

![](https://i.loli.net/2021/07/14/BA6uDE5nhg7jimL.png)

![](https://i.loli.net/2021/07/14/vlchjPZux5bLNMq.png)

使用率100%
**解决方法：**
销毁集群重新部署
销毁集群会产生用户使用权限拒绝问题
先删除/home下面tidb文件夹
然后删除tidb用户
创建tidb用户
赋予root权限
创建/home/tidb文件夹
出现下图

![](https://i.loli.net/2021/07/12/uSw27GOJRTngLHi.png)

然后
```
userdel -r tidb
```

![](https://i.loli.net/2021/07/14/jEAwcDrvq8IxQV6.png)

重新部署就可以了
部署成功