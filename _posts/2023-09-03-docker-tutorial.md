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

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904092547.png)

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904092929.png)

Build image from dockerfile

```bash
doceker build -t my-app:1.0 .
```
```bash
docker images
```

```bash
docker rm DockerconatinerID
```
``` bash
docker rmi DockerimageID
```

``` bash
docker exec -it DockerconatinerID /bin/bash
docer exec -it DockerconatinerID /bin/sh
```
终结容器，有时候 bash 不管用


### AWS
```bash
docker login
```
- 前置准备
1） 安装 AWS CLi
2） 证书设置
- 登录
- build image
- tag image
- push image

Image naming in Docker registries

registryDomain/imageName:tag
基本流程看 aws 官网

```bash
docker tag my-app:1.0 123456789012.dkr.ecr.us-east-1
```
```bash
docker push 123456789012.dkr.ecr.us-east-1
```

### 部署
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904101441.png)

### Docker Volume

3 Volume Types

- Host volumes 

你已经决定了在主机文件系统中的哪个位置进行引用。

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904102128.png)

- anonymous volumes

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904102225.png)

- named volumes

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904102350.png)

#### demo

```bash
docker-compose -f docker-compose.yaml 
```
![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904103057.png)

```bash
docker-compose -f docker-compose.yaml dowm
docker-compsae -f docker-compose.yaml up 
```

![](https://raw.githubusercontent.com/innovation64/Picimg/main/20230904103416.png)


### CPU 
Docker 限制容器 CPU 使用率的方法主要有以下三种：
1. 相对份额限制（CPU Shares）: 通过设置 --cpus 选项，可以为容器分配一个相对的 CPU 份额。例如，如果您有一个 4 核 CPU 的主机，并为某个容器分配了 2 个 CPU 份额，那么该容器将最多占用主机 CPU 的 50%。设置相对份额限制的命令如下：
  ```  
  docker run --cpus 2 -it --name my-container my-image  
  ```
2. 绝对使用限制（CPU Usage）: 通过设置 --cpu-timeout 选项，可以限制容器使用的 CPU 时间片。例如，如果您设置 --cpu-timeout 为 30 秒，则容器最多只能占用 30 秒的 CPU 时间。设置绝对使用限制的命令如下：
  ```  
  docker run --cpu-timeout 30 -it --name my-container my-image  
  ```
3. CPU 核心控制（CPU Cores）: 通过设置 --cpu 选项，可以限制容器使用的 CPU 核心数。例如，如果您有一个 4 核 CPU 的主机，并为某个容器分配了 2 个 CPU 核心，那么该容器将最多占用主机 CPU 的 50%。设置 CPU 核心控制的命令如下：
  ```  
  docker run --cpu 2 -it --name my-container my-image  
  ```
需要注意的是，以上三种限制方法可以同时使用，以实现更精确的 CPU 资源控制。

### GPU 

Docker 限制容器 GPU 使用率的方法主要有以下两种：
1. 相对份额限制（GPU Shares）: 通过设置 --gpus 选项，可以为容器分配一个相对的 GPU 份额。例如，如果您有一个具有 8 个 GPU 核心的主机，并为某个容器分配了 2 个 GPU 份额，那么该容器将最多占用主机 GPU 的 25%。设置相对份额限制的命令如下：
  ```    
  docker run --gpus 2 -it --name my-container my-image    
  ```
2. 绝对使用限制（GPU Usage）: 通过设置 --gpu-timeout 选项，可以限制容器使用的 GPU 时间片。例如，如果您设置 --gpu-timeout 为 30 秒，则容器最多只能占用 30 秒的 GPU 时间。设置绝对使用限制的命令如下：
  ```    
  docker run --gpu-timeout 30 -it --name my-container my-image    
  ```
需要注意的是，当主机有多个 GPU 时，Docker 会自动为容器分配一个默认的 GPU。如果您希望使用特定的 GPU，可以使用 --gpu 选项指定。例如，要使用主机上的第二个 GPU，可以这样设置：
```  
docker run --gpu 2 -it --name my-container my-image  
```
此外，如果您不希望限制容器的 GPU 使用，可以不设置 --gpus 和 --gpu-timeout 选项。这样，Docker 会允许容器尽可能地使用 GPU。

### 内存

Docker 限制容器内存使用率的方法主要有以下两种：
1. 限制内存总量：通过设置 --memory 选项，可以为容器分配一个有限的内存空间。例如，如果您设置 --memory 为 1G，则容器最多只能占用 1GB 的内存。设置内存总量限制的命令如下：
  ```    
  docker run --memory 1g -it --name my-container my-image    
  ```
2. 限制内存使用率：通过设置 --memory-swap 选项，可以限制容器内存使用率。例如，如果您设置 --memory-swap 为 1g，则容器最多只能占用 1GB 的内存，无论实际内存需求如何。设置内存使用率限制的命令如下：
  ```    
  docker run --memory-swap 1g -it --name my-container my-image    
  ```
需要注意的是，当您设置 --memory-swap 时，Docker 会自动为容器分配一个与 --memory 值相等的内存空间。如果您希望容器使用更多的内存，可以不设置 --memory-swap 选项。
此外，如果您希望同时限制容器的内存和 CPU 使用率，可以使用以下命令：
```    
docker run --memory 1g --cpu-timeout 30s -it --name my-container my-image    
```
在这个例子中，容器将最多占用 1GB 的内存和 30 秒的 CPU 时间片。