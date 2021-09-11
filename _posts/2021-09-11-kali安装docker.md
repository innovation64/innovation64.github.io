---
tags: kali
---
# kali安装docker
## 前言
>最近做一个控制权限的casdoor部署用docker所以需要安装

## 安装https协议、CA证书、dirmngr
```bash
apt-get update
 
apt-get install -y apt-transport-https ca-certificates
 
apt-get install dirmngr
```

## 添加GPG密钥并添加更新源
```bash
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/debian/gpg | sudo apt-key add -

echo 'deb https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/debian/ buster stable' | sudo tee /etc/apt/sources.list.d/docker.list
```

## 系统更新以及安装docker
```bash

apt-get update

apt install docker-ce
```

## 启动docker服务器和安装compose
```bash
service docker start

apt install docker-compose
```

## 测试

```bash

docker version   查看docker的版本信息
 
docker images   查看拥有的images
 
docker ps       查看docker container 
```