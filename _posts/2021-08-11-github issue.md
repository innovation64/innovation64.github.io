---
tags: github
---
# github Issue

>功能类似TODO　LIST

- feature 添加
- bug修复
- 提醒与管理
- 选择性的与某个issue关联　
  
```
commit message title, # 1
```

这个提交会作为一个comment出现在编号为１的issue的记录中
添加：
- close #n
- closes #n
- cloesd #n
- fix #n
- fixes #n
- fixed #n
- resolve #n
- resolves #n
- resolved #n
  
比如

```
commit message title, fix #n
```
则可以自动关闭第ｎ个issue

issue可以有额外的属性：
- Labels
  - enhancement
  - bug
  - invalid
  - 自定义
- Milestone 里程碑
  - 表示项目的一个阶段、比如demo、release
  - 与版本计划重合　V１.０、V２.０
  - issue不能设置截至时间但是milestone可以
- Assignee　责任人
  
充分利用可以起良好的过程管理作用
轻量级协作系统