---
tags: openEuler
---
# 环境搭建指导书
## 第一步 openEuler系统安装
openEuler系统基于Centos扩展具体安装请看官网，本人实验具体在云虚拟机中进行，实验系统配置如下：

|资源概述|详细配置|
|--|--|
|资源类型|vm|
|名称|XXX|
|开放端口|22|
|架构|鲲鹏920|
|操作系统|openEuler|
|CPU|4|
|内存|16G|
|磁盘|200G|
|网卡数量|1|
|云硬盘数量|0|
|云硬盘大小|0|

## 第二部tidb安装
具体步骤参考 [官方文档安装](https://docs.pingcap.com/zh/tidb/stable/quick-start-with-tidb#Linux)
有问题询问[社区](https://asktug.com/)
- 首先ssh连接自己的服务器
- 查看服务器配置与架构
```bash
uname -a
```
这里的服务器架构是arm64架构 显示**aarch64**
这里非常重要，因为官网默认安装的是x86的文件配置不注意这个可能报错
### 软硬件环境需求及前置检查
  - 这里官网的配置仅供参考
  - 本人没有挂在数据盘这一步省略
   ####  检测及关闭系统 swap
   ```bash
   echo "vm.swappiness = 0">> /etc/sysctl.conf
   swapoff -a && swapon -a
   sysctl -p
   ```
   #### 检测及关闭目标部署机器的防火墙
1.检查防火墙状态
```bash
 sudo firewall-cmd --state 
 sudo systemctl status firewalld.service
```
2.关闭防火墙服务
```bash
  sudo systemctl stop firewalld.service
```
3.关闭防火墙自动启动服务
```bash
sudo systemctl disable firewalld.service
```
4.检查防火墙状态
```bash
sudo systemctl status firewalld.service
```
  #### 检测及安装 NTP 服务
1.执行以下命令，如果输出 running 表示 NTP 服务正在运行：
```bash
sudo systemctl status ntpd.service
```
若返回报错信息 Unit ntpd.service could not be found.，请尝试执行以下命令，以查看与 NTP 进行时钟同步所使用的系统配置是 chronyd 还是 ntpd：
```bash
sudo systemctl status cronyd.service
```
2.执行 ntpstat 命令检测是否与 NTP 服务器同步：
openEuler系统需安装 ntpstat 软件包。
```bash
ntpstat
```
- 如果输出 synchronised to NTP server，表示正在与 NTP 服务器正常同步：
  ```
  synchronised to NTP server (85.199.214.101) at stratum 2
  time correct to within 91 ms
  polling server every 1024 s
  ```
- 以下情况表示 NTP 服务未正常同步：
  ```bash
  unsynchronised
  ```
- 以下情况表示 NTP 服务未正常运行：
  ```bash
  Unable to talk to NTP daemon. Is it running?
  ```
如果要在 openEuler 系统上手动安装 NTP 服务，可执行以下命令：
```bash
sudo yum install ntp ntpdate && \
sudo systemctl start ntpd.service && \
sudo systemctl enable ntpd.service
```
### 在 Linux OS 上部署本地测试环境
>TiDB 是一个分布式系统。最基础的 TiDB 测试集群通常由 2 个 TiDB 实例、3 个 TiKV 实例、3 个 PD 实例和可选的 TiFlash 实例构成。通过 TiUP Playground，可以快速搭建出上述的一套基础测试集群。
**本测试近在单机模拟所以用不上TiFlash所以在yaml文件中注释掉了tiflash相关内容防止启动出错**

1.下载并安装 TiUP
```bash
curl --proto '=https' --tlsv1.2 -sSf https://tiup-mirrors.pingcap.com/install.sh | sh
```
2.声明全局环境变量
```
source .bash_profile
```
>TiUP 主要通过以下一些命令来管理组件：

- list：查询组件列表，用于了解可以安装哪些组件，以及这些组件可选哪些版本
- install：安装某个组件的特定版本
- update：升级某个组件到最新的版本
- uninstall：卸载组件
- status：查看组件运行状态
- clean：清理组件实例
- help：打印帮助信息，后面跟其他 TiUP 命令则是打印该命令的使用方法

### 使用 TiUP cluster 在单机上模拟生产环境部署步骤
>适用场景：希望用单台 Linux 服务器，体验 TiDB 最小的完整拓扑的集群，并模拟生产的部署步骤。
>耗时：10 分钟
本节介绍如何参照 TiUP 最小拓扑的一个 YAML 文件部署 TiDB 集群。
3.安装 TiUP 的 cluster 组件：
```bash
tiup cluster
```
4.如果机器已经安装 TiUP cluster，需要更新软件版本：
```
tiup update --self && tiup update cluster
```
5.由于模拟多机部署，需要通过 root 用户调大 sshd 服务的连接数限制：
i:修改 /etc/ssh/sshd_config 将**MaxSessions** 调至 20。
ii:重启 sshd 服务：
```bash
service sshd restart
```
6.创建并启动集群
按下面的配置模板，编辑配置文件，命名为 topo.yaml，其中：
```yaml
global:
 arch: "arm64"
 user: "tidb"
 ssh_port: 22
 deploy_dir: "/tidb-deploy"
 data_dir: "/tidb-data"

# # Monitored variables are applied to all the machines.
monitored:
 node_exporter_port: 9100
 blackbox_exporter_port: 9115

server_configs:
 tidb:
   log.slow-threshold: 300
   binlog.enable: false
   binlog.ignore-error: false

 tikv:
   readpool.storage.use-unified-pool: false
   readpool.coprocessor.use-unified-pool: true
 pd:
   replication.enable-placement-rules: true
   replication.location-labels: ["host"]
 tiflash:
   logger.level: "info"

pd_servers:
 - host: 14.0.0.130

tidb_servers:
 - host: 14.0.0.130

tikv_servers:
 - host: 14.0.0.130
   port: 20160
   status_port: 20180
   config:
     server.labels: { host: "logic-host-1" }

 - host: 14.0.0.130
   port: 20161
   status_port: 20181
   config:
     server.labels: { host: "logic-host-2" }

 - host: 14.0.0.130
   port: 20162
   status_port: 20182
   config:
     server.labels: { host: "logic-host-3" }

#tiflash_servers:
# - host: 14.0.0.130

monitoring_servers:
 - host: 14.0.0.130

grafana_servers:
 - host: 14.0.0.130

```
7.执行集群部署命令：
```bash
tiup cluster deploy <cluster-name> <tidb-version> ./topo.yaml --user root -p
```

- 参数 **<cluster-name/.>** 表示设置集群名称
- 参数 <tidb-version> 表示设置集群版本，可以通过 tiup list tidb 命令来查看当前支持部署的 TiDB 版本
- **这里选用v5.1.0**
按照引导，输入”y”及 root 密码，来完成部署：
```bash
Do you want to continue? [y/N]:  y
Input SSH password:
```
8.启动集群：
```bash
tiup cluster start <cluster-name>
```
9.访问集群：
- 安装 MySQL 客户端。如果已安装 MySQL 客户端则可跳过这一步骤。
```bash
yum -y install mysql
```
- 建立软连接
```bash
ln -s /usr/local/mysql/bin/mysql /usr/bin
```
- 访问 TiDB 数据库，密码为空：
```bash
mysql -h 14.0.0.130 -P 4000 -u root
```
访问 TiDB 的 Grafana 监控：

通过 http://{grafana-ip}:3000 访问集群 Grafana 监控页面，默认用户名和密码均为 admin。

访问 TiDB 的 Dashboard：

通过 http://{pd-ip}:2379/dashboard 访问集群 TiDB Dashboard 监控页面，默认用户名为 root，密码为空。

执行以下命令确认当前已经部署的集群列表：

```tiup cluster list```
执行以下命令查看集群的拓扑结构和状态：

```tiup cluster display <cluster-name>```
![](https://i.loli.net/2021/07/13/fWAd8CtbKZ2OrQ3.png)
## benchmark环境搭建
TPC-C 是一个对 OLTP（联机交易处理）系统进行测试的规范，使用一个商品销售模型对 OLTP 系统进行测试，其中包含五类事务：

- NewOrder – 新订单的生成
- Payment – 订单付款
- OrderStatus – 最近订单查询
- Delivery – 配送
- StockLevel – 库存缺货状态分析
在测试开始前，TPC-C Benchmark 规定了数据库的初始状态，也就是数据库中数据生成的规则，其中 ITEM 表中固定包含 10 万种商品，仓库的数量可进行调整，假设 WAREHOUSE 表中有 W 条记录，那么：
- STOCK 表中应有 W * 10 万条记录（每个仓库对应 10 万种商品的库存数据）
- DISTRICT 表中应有 W * 10 条记录（每个仓库为 10 个地区提供服务）
- CUSTOMER 表中应有 W * 10 * 3000 条记录（每个地区有 3000 个客户）
- HISTORY 表中应有 W * 10 * 3000 条记录（每个客户一条交易历史）
- ORDER 表中应有 W * 10 * 3000 条记录（每个地区 3000 个订单），并且最后生成的 900 个订单被添加到 NEW-ORDER 表中，每个订单随机生成 5 ~ 15 条 ORDER-LINE 记录。

我们将以 1000 WAREHOUSE 为例进行测试。

TPC-C 使用 tpmC 值（Transactions per Minute）来衡量系统最大有效吞吐量 (MQTh, Max Qualified Throughput)，其中 Transactions 以 NewOrder Transaction 为准，即最终衡量单位为每分钟处理的新订单数。

本文使用 go-tpc 作为 TPC-C 测试实现，可以通过 TiUP 命令下载测试程序:
```bash
tiup install bench
```
导入数据
导入数据通常是整个 TPC-C 测试中最耗时，也是最容易出问题的阶段。

在 shell 中运行 TiUP 命令：
```bash
tiup bench tpcc -H 172.16.5.140 -P 4000 -D tpcc --warehouses 1000 prepare
```
运行测试
运行测试的命令是：
```bash
tiup bench tpcc -H 172.16.5.140 -P 4000 -D tpcc --warehouses 1000 run
```
清理测试数据
```bash
tiup bench tpcc -H 172.16.5.140 -P 4000 -D tpcc --warehouses 4 cleanup
```