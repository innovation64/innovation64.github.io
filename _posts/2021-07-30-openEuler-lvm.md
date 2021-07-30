---
tags: openEuler
---
# lvm磁盘划分
## lvm
### 物理存储介质（The physical media）：这里指系统的存储设备：硬盘，如：/dev/xvda、/dev/vdb等等，是存储系统最低层的存储单元。

### PV（Physical Volume）- 物理卷
>物理卷在逻辑卷管理中处于最底层，它可以是实际物理硬盘上的分区，也可以是整个物理硬盘

### VG（Volumne Group）- 卷组
>卷组建立在物理卷之上，一个卷组中至少要包括一个物理卷，在卷组建立之后可动态添加物理卷到卷组中。一个逻辑卷管理系统工程中可以只有一个卷组，也可以拥有多个卷组。

### LV（Logical Volume）- 逻辑卷
>逻辑卷建立在卷组之上，卷组中的未分配空间可以用于建立新的逻辑卷，逻辑卷建立后可以动态地扩展和缩小空间。系统中的多个逻辑卷可以属于同一个卷组，也可以属于不同的多个卷组。
### LVM使用分层结构，如下图所示：
![](https://i.loli.net/2021/07/30/8CA4zkJFa9q7hOr.png)

## 使用
首先检查是否安装LVM管理工具

然后是扩容问题

我申请的是磁盘容量为200G的虚拟机，但是系统使用只有78G，所以需要扩容未划分的磁盘，相比于磁盘已满的需要挂载的区别就在多一步挂在磁盘，然后分区时候注意路径

## 扩容
- 查看已经安装
```bash
fdisk -l
```
![](https://i.loli.net/2021/07/30/7guVR8UdYBcxWDw.png)

- 查看其附属结构

```bash
lsblk
```
![](https://i.loli.net/2021/07/30/p4icQ1OrBDqmefu.png)

- 查看具体分区
```bash
df -h
```
![](https://i.loli.net/2021/07/30/HoZquvdJPpMgR4N.png)

- 查看lv
```bash
lvdispaly
```
![](https://i.loli.net/2021/07/30/lq13Pg4bVwSvBTJ.png)
![](https://i.loli.net/2021/07/30/f9qdMBeuVyTzGRt.png)

- 查看vg
```bash
vgdisplay
```
![](https://i.loli.net/2021/07/30/FmTYM3zfZqIgbKG.png)

- 查看pv
```bash
pvdisplay
```
![](https://i.loli.net/2021/07/30/r9TM4NzwnxEIovW.png)

- 查看具体路径
```bash
fdisk -l | grep 'dev'
```
![](https://i.loli.net/2021/07/30/lsJfF6bQvZ1yn7r.png)

- 查看vg名字
```bash
pvsacn
```
![](https://i.loli.net/2021/07/30/e2fZJEqVXPWvRuh.png)

- 扩容
```bash
fdisk /dev/sda
```
![](https://i.loli.net/2021/07/30/zdcmtqeN54gaDGR.png)

```bash
vgextend openeuler /dev/sda4
```
![](https://i.loli.net/2021/07/30/4mi6uBFAsfWLGwO.png)

```bash
lvextend -l +100%FREE /dev/mapper/openeuler-root
```
## 更新记录
```bash
e2fsck -f /dev/mapper/openeuler-root
resize2fs /dev/mapper/openeuler-root
```
## 扩容完成
![](https://i.loli.net/2021/07/30/ip7fZBGUsgPuRam.png)




