---
tags: docker
---
# Docker 从零开始入门到入土
## 什么是容器
> - 一种打包技术，将软件运行环境和所依赖的所有东西打包在一起，可以实现软件的跨平台 
> - 便携，容易分享与移动 
> - 使得更高效的开发与部署

## 容器在哪里？
- 容器仓库
- 私有仓库
- Docker的公共仓库 在 dockerhub

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903214732.png)

## 容器怎么提高开发效率

|之前|容器后|
|--|--|
|在每个操作系统安装步骤不同 |拥有独立环境直接打包所有配置|
|不知道那一步错了|一键安装应用可以同时跑两个不同版本|
|需要服务器设置并且要说明书|开发和运营共同打包到一个容器，服务器端不需要环境设置除了 Docker Runtime|
|![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903215000.png)|![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903215223.png)|
|![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903215754.png)|![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903215829.png)|

container

- 镜像层
- 大部分是基于 linux 的镜像 ，因为很小 
- 最火的是应用镜像

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903220339.png)

# instructors

```bash
docker ps
```

```bash
docker run xxx:xx.xx
```

#### 镜像与容器的区别
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903220926.png)

## docker 和 虚拟机区别
- Docker 在操作系统级别
- 不同等级的抽象
- 为什么基于 Linux 的 docker 容器不能运行在 windows 上

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903221417.png)

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903221543.png)

## 安装 Docker
Docker Toolbox 针对老式操作系统

两个版本 CE or EE

个人社区版本足够

### MAC

至少有 4G RAM

### WINDOWS
确认可虚拟化

### Linux
要64位
ubuntu 支持 x86_64 armhf s390x ppc64le
- 设置仓库
- 手动安装包

看官网命令
```
sudo apt-get update
sudo apt-get install docker-ce
sudo docker run hello-world
```
### Docker Toolbox
官网直接下载对应版本

## 基本命令

### 镜像与容器的区别
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903223809.png)

```bash
docker ps 
```
列出运行的容器

```bash
docker stop containerID
```
停止容器

```bash
docker start containerID
```
启动容器

```bash
docker run -d xxx.xxx
```
拉去镜像并创建容器

### 容器端口与主机端口

- 多个容器可以运行在你的主机
- 你的电脑只有部分端口可以使用
- 主机端口冲突时
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903224808.png)

```bash
docker run -p 80:8080 xxx.xxx
```

## debugging docker

```bash
docker logs containerID 
```
```bash
docker exec -it containerID bash
```

## demo
### workflow with docker(工作流)

这里以开发 JS APP 为例
JS + MongoDB

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903230516.png)

### 开发
JS + nodejs
SD 

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903230729.png)

```bash
docker pull mongo
docker pull mongo-express
```

```bash
docker network ls
docker network create mongo-net
```
```bash
docker run -p 27017:27017 --name mongodb -d mongo -e MONGODB_INITDB_ROOT_USER=admin -e MONGODB_INITDB_ROOT_PASSWORD=password --net mongo-net
```
```bash
docker logs
```

```bash
docker run -d \ -p 8081:8081\ -e ME_CONFIG_MONGODB_ADMINUSERNAME=admin\ -e ME_CONFIG_MONGODB_ADMINPASSWORD=password\ --name mongo-express\ --net mongo-net\ -e ME_CONFIG_MONGODB_SERVER=mongodb\ mongo-express
```

```bash
docker logs XXXXX
```

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903232943.png)

## docker compose

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903233244.png)

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230903233335.png)


### 创建 docker compose 文件

```bash
docker-compose -f mongo.yaml up
```

### dockerfile
