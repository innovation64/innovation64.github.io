---
tags: openEuler
---

# 磁盘划分错误
## GPT PMBR size mismatch (167772159 != 419430399) will be corrected by write.
The backup GPT table is not on the end of the device. This problem will be corrected by write.
```bash
sudo parted -l
```
## 磁盘划分
```bash
fdisk -l
```
显示分区
![](https://i.loli.net/2021/07/26/CIJES8BD3fYPvym.png)