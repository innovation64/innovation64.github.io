---
tags: openEuler
---
# Linux卸载指南
## CentOS系统使用yum卸载软件包
>任何Linux发行版都包含许多的软件包，为了管理这些软件包，必须有合适的软件包管理器。根据Linux发行版使用的软件包类型的不同，软件包管理器也会有所不同。使用RPM软件包的Linux发行版（CentOS、RHEL、Fedora和OpenSUSE等），使用的是yum软件包管理器。本文我们来介绍一下，如何卸载yum软件包。

## 使用 yum 卸载软件包
### 列出已安装的软件包
可以使用`grep`过滤输出内容。我们使用如下命令

```bash
yum list installed | grep <search_term>
```

使用less查看已安装的软件包列表，可以通过分页的方式来展现：

```bash
yum list installed | less
```
### 基本的软件包卸载

```bash
yum remove <package>
```

**如果想要卸载多个软件包，则可以使用下面的语法结构：**

```bash
yum remove <package_1> <package_2>
```
### 卸载软件包组

查看该软件包组的信息，可以运行以下命令：

```bash
 yum groupinfo Development Tools
```

如果我们要卸载该名为“Development Tools”的软件包组，可以运行如下命令：

```bash
yum remove @"Development Tools"
```

**也可以执行如下的命令来卸载：**

```bash
yum group remove "<group_name>"
```
## RMP包管理
### 1.rpm包的管理
>介绍：
>一种用于互联网下载包的打包及安装工具，它包含在某些Linux分发版中，它生成具有RPM扩展名的文件，RPM是RedHat Package Manager（RedHat软件包管理工具）的缩写，类似windows的setup.exe，这一文件格式名称虽然打上了RedHat的标志，但理念是通用的
>Linux的分发版本都有采用（suse,redhat, centos 等等），可以算是公认的行业标准了

### 2.rpm包的简单查询指令：
查询已安装的rpm列表 rpm  –qa | grep xx（q表示query，a表示查询所有，grep表示过滤）

例如：查询Linux中是否安装有firefox

```bash
rpm -qa | grep firefox
```

rpm包名基本格式：
一个rpm包名：firefox-45.0.1-1.el6.centos.x86_64.rpm

名称:firefox

版本号：45.0.1-1

适用操作系统: el6.centos.x86_64 表示centos6.x的64位系统

如果是i686、i386表示32位系统，noarch表示通用

#### rpm包的其它查询指令：
- rpm -qa：查询所安装的所有rpm软件包

- rpm -qa | more ：查询所安装的所有rpm软件包 并且分页显示

- rpm -qa | grep X [rpm -qa | grep firefox ] ：查询是否安装有某个软件（火狐的软件）

- rpm -q 软件包名 ：查询软件包是否安装 rpm -q firefox
- rpm -qi 软件包名 ：查询软件包信息

- rpm -ql 软件包名 ：查询软件包中的文件的安装位置

- rpm -qf 文件全路径名：查询文件所属的软件包 ，例如：rpm -qf /etc/passwd 

### 3.卸载rpm包：

基本语法 rpm -e RPM包的名称 

应用案例 ：删除firefox  软件包 

```bash
rpm -e firefox
```

细节讨论：

(1) 如果其它软件包依赖于您要卸载的软件包，卸载时则会产生错误信息

如： $ rpm -e foo removing these packages would break dependencies:foo is needed by bar-1.0-1

(2) 如果我们就是要删除 foo这个rpm 包，可以增加参数 --nodeps ,就可以强制删除，但是一般不推荐这样做，因为依赖于该软件包的程序可能无法运行 如：$ rpm -e --nodeps foo 

### 4.安装rpm包

 基本语法： rpm -ivh  RPM包全路径名称

 参数说明： i=install 安装 v=verbose 提示 h=hash  进度条

 应用实例： 安装firefox浏览器

 ① 先要找到安装包
 >需要先挂载上我们之前安装CentOS的iso文件，点击虚拟机设置使用ios文件，那么在系统中会多出一个光驱，打开命令行终端进入到/media/CentOS_6.8_final/package/
>（所有的rpm包都在这里） 
>点击然后拷贝到/opt目录下  cp firefox（ rpm包的文件名） 
> 需要拷贝到的目录 （可以输入rpm包名字的前几个字母然后使用Tab键补全）

②安装

>切换到/opt目录下找到刚才拷贝的文件然后使用： rpm ivh + 火狐rpm软件安装包的名字就可以了

## 用终端卸载本机Ubuntu应用程序
终端快捷方式
`Ctrl + Alt + T`
### 列出所有已经安装的应用

```bash
dpkg --list
```

或者

```bash
sudo apt --installed list | more
```

使用以下格式。 您必须替换“包裹名字'在示例中按实际要卸载的软件包名称：

```bash
sudo apt-get remove nombre-del -paquete
```

这将从我们的系统中删除该应用程序，但保留配置文件，插件和设置以供将来使用。 如果我们要 从我们的系统中完全删除该应用程序，我们还将使用以下命令:

```bash
sudo apt-get purge nombre-paquete
```

### 使用终端卸载Snap软件包
我们还可以使用终端（Ctrl + Alt + T）删除已安装的snap软件包。 首先，我们可以 全部列出 执行以下命令：

```bash
snap list
```

找到要移除的包裹后，在同一终端中，我们只有 使用以下语法:

```bash
sudo snap remove nombre-del-paquete

```
我们只需要替换“包裹名字'按快照应用程序的实际程序包名称。

### 使用终端卸载Flatpak应用程序
如果您通过flatpak安装了应用程序，则还可以使用终端将其删除。 首先得到 表Flatpak套餐 已安装 在终端（Ctrl + Alt + T）中运行以下命令：

```bash
flatpak list
```
找到要卸载的flatpak应用程序后，您只需 请按照以下语法删除应用程序:

```bash
sudo flatpak uninstall nombre-del-paquete

```
与之前的选项一样，您必须替换“包裹名字'以flatpak应用的名称命名。

### Ubuntu apt-get卸载软件包
操作步骤：

>1.sudo apt-get autoremove（卸载系统中所有未被使用的依赖关系）

>2.sudo apt-get clean（清除所有缓存的包文件）

以上操作绿色无害，对系统无影响。

apt-get的卸载相关的命令有remove/purge/autoremove/clean/autoclean等。具体来说：

```bash
apt-get purge / apt-get --purge remove
```

删除已安装包（不保留配置文件)。
如软件包a，依赖软件包b，则执行该命令会删除a，而且不保留配置文件

```bash
apt-get autoremove
```

删除为了满足依赖而安装的，但现在不再需要的软件包（包括已安装包），保留配置文件。

```bash
apt-get remove
```

删除已安装的软件包（保留配置文件），不会删除依赖软件包，且保留配置文件。

```bash
apt-get autoclean
```

APT的底层包是dpkg, 而dpkg 安装Package时, 会将 *.deb 放在 /var/cache/apt/archives/中，apt-get autoclean 只会删除 /var/cache/apt/archives/ 已经过期的deb。

```bash
apt-get clean
```

使用 apt-get clean 会将 /var/cache/apt/archives/ 的 所有 deb 删掉，可以理解为 rm /var/cache/apt/archives/*.deb。

那么如何彻底卸载软件呢？
具体来说可以运行如下命令：

#### 删除软件及其配置文件
```bash
apt-get --purge remove
```

#### 删除没用的依赖包
```bash
apt-get autoremove
```

#### 此时dpkg的列表中有“rc”状态的软件包，可以执行如下命令做最后清理：
```bash
dpkg -l |grep ^rc|awk '{print $2}' |sudo xargs dpkg -P
```

当然如果要删除暂存的软件安装包，也可以再使用clean命令。