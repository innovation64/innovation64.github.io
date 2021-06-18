---
tags: 数据分析
---
### Anaconda简介
*本教程来源于学堂在线*
- [安装方式参考](https://zhuanlan.zhihu.com/p/32925500)

- [官方下载地址](https://www.anaconda.com/products/individual)

### MacOS下部分命令

- 创建名为ml的python3.6环境 `conda create -n ml python=3.6`

- 创建名为ml的环境 `conda env remove -n ml`

- 进入名为ml的环境 `source activate ml`

- 退出环境 `conda deactivate`

- 安装工具包 `conda install numpy pandas scikit-learn`

- 查看所有环境名字 `conda env list`

- 查看当前环境下所有已安装的工具包 `conda list`

### Pandas & Numpy

- Pandas：一个快速高效的**数据分析**库，[中文官网](https://www.pypandas.cn/)，[英文官网](https://pandas.pydata.org/)
- numpy：一个快速高效的**数学运算**库，[中文官网](https://www.numpy.org.cn/)，[英文官网](https://numpy.org/)

为帮助python基础较弱的同学完成案例作业，下面为大家演示讲解numpy和pandas里的一些常用函数，了解基本操作方便之后使用。

### 载入工具包


```python
import pandas as pd
import numpy as np
```

# Pandas
### 初始化一个pandas的DataFrame

比较常用的一种初始化方式是从python的字典初始化，列名是字典的key，每列的元素是字典的value，要求每列长度相同


```python
df = pd.DataFrame({'age': [1,2,3], 'name': ['a', 'b', 'c']})
print(df)
print(type(df))
```

       age name
    0    1    a
    1    2    b
    2    3    c
    <class 'pandas.core.frame.DataFrame'>


pandas也可以读入一个csv文件成DataFrame


```python
df = pd.read_csv('./data/high_diamond_ranked_10min.csv', sep=',')
print(type(df))
```

    <class 'pandas.core.frame.DataFrame'>


### 查看DataFrame的信息


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9879 entries, 0 to 9878
    Data columns (total 40 columns):
    gameId                          9879 non-null int64
    blueWins                        9879 non-null int64
    blueWardsPlaced                 9879 non-null int64
    blueWardsDestroyed              9879 non-null int64
    blueFirstBlood                  9879 non-null int64
    blueKills                       9879 non-null int64
    blueDeaths                      9879 non-null int64
    blueAssists                     9879 non-null int64
    blueEliteMonsters               9879 non-null int64
    blueDragons                     9879 non-null int64
    blueHeralds                     9879 non-null int64
    blueTowersDestroyed             9879 non-null int64
    blueTotalGold                   9879 non-null int64
    blueAvgLevel                    9879 non-null float64
    blueTotalExperience             9879 non-null int64
    blueTotalMinionsKilled          9879 non-null int64
    blueTotalJungleMinionsKilled    9879 non-null int64
    blueGoldDiff                    9879 non-null int64
    blueExperienceDiff              9879 non-null int64
    blueCSPerMin                    9879 non-null float64
    blueGoldPerMin                  9879 non-null float64
    redWardsPlaced                  9879 non-null int64
    redWardsDestroyed               9879 non-null int64
    redFirstBlood                   9879 non-null int64
    redKills                        9879 non-null int64
    redDeaths                       9879 non-null int64
    redAssists                      9879 non-null int64
    redEliteMonsters                9879 non-null int64
    redDragons                      9879 non-null int64
    redHeralds                      9879 non-null int64
    redTowersDestroyed              9879 non-null int64
    redTotalGold                    9879 non-null int64
    redAvgLevel                     9879 non-null float64
    redTotalExperience              9879 non-null int64
    redTotalMinionsKilled           9879 non-null int64
    redTotalJungleMinionsKilled     9879 non-null int64
    redGoldDiff                     9879 non-null int64
    redExperienceDiff               9879 non-null int64
    redCSPerMin                     9879 non-null float64
    redGoldPerMin                   9879 non-null float64
    dtypes: float64(6), int64(34)
    memory usage: 3.0 MB


### 查看头/尾10条数据


```python
# df.head(10)
# df.tail(10)
print(df)
print(df.drop(columns=['gameId']))
print(df)
```

              gameId  blueWins  blueWardsPlaced  blueWardsDestroyed  \
    0     4519157822         0               28                   2   
    1     4523371949         0               12                   1   
    2     4521474530         0               15                   0   
    3     4524384067         0               43                   1   
    4     4436033771         0               75                   4   
    ...          ...       ...              ...                 ...   
    9874  4527873286         1               17                   2   
    9875  4527797466         1               54                   0   
    9876  4527713716         0               23                   1   
    9877  4527628313         0               14                   4   
    9878  4523772935         1               18                   0   
    
          blueFirstBlood  blueKills  blueDeaths  blueAssists  blueEliteMonsters  \
    0                  1          9           6           11                  0   
    1                  0          5           5            5                  0   
    2                  0          7          11            4                  1   
    3                  0          4           5            5                  1   
    4                  0          6           6            6                  0   
    ...              ...        ...         ...          ...                ...   
    9874               1          7           4            5                  1   
    9875               0          6           4            8                  1   
    9876               0          6           7            5                  0   
    9877               1          2           3            3                  1   
    9878               1          6           6            5                  0   
    
          blueDragons  ...  redTowersDestroyed  redTotalGold  redAvgLevel  \
    0               0  ...                   0         16567          6.8   
    1               0  ...                   1         17620          6.8   
    2               1  ...                   0         17285          6.8   
    3               0  ...                   0         16478          7.0   
    4               0  ...                   0         17404          7.0   
    ...           ...  ...                 ...           ...          ...   
    9874            1  ...                   0         15246          6.8   
    9875            1  ...                   0         15456          7.0   
    9876            0  ...                   0         18319          7.4   
    9877            1  ...                   0         15298          7.2   
    9878            0  ...                   0         15339          6.8   
    
          redTotalExperience  redTotalMinionsKilled  redTotalJungleMinionsKilled  \
    0                  17047                    197                           55   
    1                  17438                    240                           52   
    2                  17254                    203                           28   
    3                  17961                    235                           47   
    4                  18313                    225                           67   
    ...                  ...                    ...                          ...   
    9874               16498                    229                           34   
    9875               18367                    206                           56   
    9876               19909                    261                           60   
    9877               18314                    247                           40   
    9878               17379                    201                           46   
    
          redGoldDiff  redExperienceDiff  redCSPerMin  redGoldPerMin  
    0            -643                  8         19.7         1656.7  
    1            2908               1173         24.0         1762.0  
    2            1172               1033         20.3         1728.5  
    3            1321                  7         23.5         1647.8  
    4            1004               -230         22.5         1740.4  
    ...           ...                ...          ...            ...  
    9874        -2519              -2469         22.9         1524.6  
    9875         -782               -888         20.6         1545.6  
    9876         2416               1877         26.1         1831.9  
    9877          839               1085         24.7         1529.8  
    9878         -927                 58         20.1         1533.9  
    
    [9879 rows x 40 columns]
          blueWins  blueWardsPlaced  blueWardsDestroyed  blueFirstBlood  \
    0            0               28                   2               1   
    1            0               12                   1               0   
    2            0               15                   0               0   
    3            0               43                   1               0   
    4            0               75                   4               0   
    ...        ...              ...                 ...             ...   
    9874         1               17                   2               1   
    9875         1               54                   0               0   
    9876         0               23                   1               0   
    9877         0               14                   4               1   
    9878         1               18                   0               1   
    
          blueKills  blueDeaths  blueAssists  blueEliteMonsters  blueDragons  \
    0             9           6           11                  0            0   
    1             5           5            5                  0            0   
    2             7          11            4                  1            1   
    3             4           5            5                  1            0   
    4             6           6            6                  0            0   
    ...         ...         ...          ...                ...          ...   
    9874          7           4            5                  1            1   
    9875          6           4            8                  1            1   
    9876          6           7            5                  0            0   
    9877          2           3            3                  1            1   
    9878          6           6            5                  0            0   
    
          blueHeralds  ...  redTowersDestroyed  redTotalGold  redAvgLevel  \
    0               0  ...                   0         16567          6.8   
    1               0  ...                   1         17620          6.8   
    2               0  ...                   0         17285          6.8   
    3               1  ...                   0         16478          7.0   
    4               0  ...                   0         17404          7.0   
    ...           ...  ...                 ...           ...          ...   
    9874            0  ...                   0         15246          6.8   
    9875            0  ...                   0         15456          7.0   
    9876            0  ...                   0         18319          7.4   
    9877            0  ...                   0         15298          7.2   
    9878            0  ...                   0         15339          6.8   
    
          redTotalExperience  redTotalMinionsKilled  redTotalJungleMinionsKilled  \
    0                  17047                    197                           55   
    1                  17438                    240                           52   
    2                  17254                    203                           28   
    3                  17961                    235                           47   
    4                  18313                    225                           67   
    ...                  ...                    ...                          ...   
    9874               16498                    229                           34   
    9875               18367                    206                           56   
    9876               19909                    261                           60   
    9877               18314                    247                           40   
    9878               17379                    201                           46   
    
          redGoldDiff  redExperienceDiff  redCSPerMin  redGoldPerMin  
    0            -643                  8         19.7         1656.7  
    1            2908               1173         24.0         1762.0  
    2            1172               1033         20.3         1728.5  
    3            1321                  7         23.5         1647.8  
    4            1004               -230         22.5         1740.4  
    ...           ...                ...          ...            ...  
    9874        -2519              -2469         22.9         1524.6  
    9875         -782               -888         20.6         1545.6  
    9876         2416               1877         26.1         1831.9  
    9877          839               1085         24.7         1529.8  
    9878         -927                 58         20.1         1533.9  
    
    [9879 rows x 39 columns]
              gameId  blueWins  blueWardsPlaced  blueWardsDestroyed  \
    0     4519157822         0               28                   2   
    1     4523371949         0               12                   1   
    2     4521474530         0               15                   0   
    3     4524384067         0               43                   1   
    4     4436033771         0               75                   4   
    ...          ...       ...              ...                 ...   
    9874  4527873286         1               17                   2   
    9875  4527797466         1               54                   0   
    9876  4527713716         0               23                   1   
    9877  4527628313         0               14                   4   
    9878  4523772935         1               18                   0   
    
          blueFirstBlood  blueKills  blueDeaths  blueAssists  blueEliteMonsters  \
    0                  1          9           6           11                  0   
    1                  0          5           5            5                  0   
    2                  0          7          11            4                  1   
    3                  0          4           5            5                  1   
    4                  0          6           6            6                  0   
    ...              ...        ...         ...          ...                ...   
    9874               1          7           4            5                  1   
    9875               0          6           4            8                  1   
    9876               0          6           7            5                  0   
    9877               1          2           3            3                  1   
    9878               1          6           6            5                  0   
    
          blueDragons  ...  redTowersDestroyed  redTotalGold  redAvgLevel  \
    0               0  ...                   0         16567          6.8   
    1               0  ...                   1         17620          6.8   
    2               1  ...                   0         17285          6.8   
    3               0  ...                   0         16478          7.0   
    4               0  ...                   0         17404          7.0   
    ...           ...  ...                 ...           ...          ...   
    9874            1  ...                   0         15246          6.8   
    9875            1  ...                   0         15456          7.0   
    9876            0  ...                   0         18319          7.4   
    9877            1  ...                   0         15298          7.2   
    9878            0  ...                   0         15339          6.8   
    
          redTotalExperience  redTotalMinionsKilled  redTotalJungleMinionsKilled  \
    0                  17047                    197                           55   
    1                  17438                    240                           52   
    2                  17254                    203                           28   
    3                  17961                    235                           47   
    4                  18313                    225                           67   
    ...                  ...                    ...                          ...   
    9874               16498                    229                           34   
    9875               18367                    206                           56   
    9876               19909                    261                           60   
    9877               18314                    247                           40   
    9878               17379                    201                           46   
    
          redGoldDiff  redExperienceDiff  redCSPerMin  redGoldPerMin  
    0            -643                  8         19.7         1656.7  
    1            2908               1173         24.0         1762.0  
    2            1172               1033         20.3         1728.5  
    3            1321                  7         23.5         1647.8  
    4            1004               -230         22.5         1740.4  
    ...           ...                ...          ...            ...  
    9874        -2519              -2469         22.9         1524.6  
    9875         -782               -888         20.6         1545.6  
    9876         2416               1877         26.1         1831.9  
    9877          839               1085         24.7         1529.8  
    9878         -927                 58         20.1         1533.9  
    
    [9879 rows x 40 columns]


### DataFrame每行的index
这里的index就是普通的数字0-9879，有时index也可以是其他一些特殊对象，如日期时间等。


```python
print(df.index)
```

    RangeIndex(start=0, stop=9879, step=1)


### 列名
csv文件一般列名读入是字符串str，有些情况下没有列名或简单以数字作为列名。


```python
print(df.columns)
cols = df.columns[:2]
print(type(cols))
cols = list(cols) + ['blueAvgLevel', 'redAvgLevel']
print(cols)
```

    Index(['gameId', 'blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
           'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
           'blueEliteMonsters', 'blueDragons', 'blueHeralds',
           'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
           'blueTotalExperience', 'blueTotalMinionsKilled',
           'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
           'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
           'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
           'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
           'redTotalGold', 'redAvgLevel', 'redTotalExperience',
           'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
           'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'],
          dtype='object')
    <class 'pandas.core.indexes.base.Index'>
    ['gameId', 'blueWins', 'blueAvgLevel', 'redAvgLevel']


### 访问列
可以传入列名字符串（返回Series）或列名的list（返回DataFame）


```python
df['redAvgLevel']
# df[cols]
type(df[['redAvgLevel']])
```




    pandas.core.frame.DataFrame



### 访问行


```python
df[0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>...</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4519157822</td>
      <td>0</td>
      <td>28</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16567</td>
      <td>6.8</td>
      <td>17047</td>
      <td>197</td>
      <td>55</td>
      <td>-643</td>
      <td>8</td>
      <td>19.7</td>
      <td>1656.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4523371949</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>17620</td>
      <td>6.8</td>
      <td>17438</td>
      <td>240</td>
      <td>52</td>
      <td>2908</td>
      <td>1173</td>
      <td>24.0</td>
      <td>1762.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4521474530</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>17285</td>
      <td>6.8</td>
      <td>17254</td>
      <td>203</td>
      <td>28</td>
      <td>1172</td>
      <td>1033</td>
      <td>20.3</td>
      <td>1728.5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 40 columns</p>
</div>



### 访问某个区域
例如前3行，cols对应的4列


```python
df.loc[range(0,3), cols]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueAvgLevel</th>
      <th>redAvgLevel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4519157822</td>
      <td>0</td>
      <td>6.6</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4523371949</td>
      <td>0</td>
      <td>6.6</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4521474530</td>
      <td>0</td>
      <td>6.4</td>
      <td>6.8</td>
    </tr>
  </tbody>
</table>
</div>



不建议在loc中使用形如0:3这样的index，因为行为略为反常。


```python
df.loc[0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>...</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4519157822</td>
      <td>0</td>
      <td>28</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16567</td>
      <td>6.8</td>
      <td>17047</td>
      <td>197</td>
      <td>55</td>
      <td>-643</td>
      <td>8</td>
      <td>19.7</td>
      <td>1656.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4523371949</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>17620</td>
      <td>6.8</td>
      <td>17438</td>
      <td>240</td>
      <td>52</td>
      <td>2908</td>
      <td>1173</td>
      <td>24.0</td>
      <td>1762.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4521474530</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>17285</td>
      <td>6.8</td>
      <td>17254</td>
      <td>203</td>
      <td>28</td>
      <td>1172</td>
      <td>1033</td>
      <td>20.3</td>
      <td>1728.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4524384067</td>
      <td>0</td>
      <td>43</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16478</td>
      <td>7.0</td>
      <td>17961</td>
      <td>235</td>
      <td>47</td>
      <td>1321</td>
      <td>7</td>
      <td>23.5</td>
      <td>1647.8</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 40 columns</p>
</div>



### 访问某个元素
loc是传入index和列名，
但是iloc传入的是编号，无论index和column是否为数字，都传入0-xxx的数字下标


```python
df.loc[0, 'gameId']
df.iloc[0, 0]
df.at[0, 'gameId']
```




    4519157822



### 拷贝
=是引用一个DataFrame对象，修改df_copy则df也会发生变化。
如果不想原df被修改，可以使用copy深拷贝一个DataFrame对象。


```python
df_copy = df
df_copy = df.copy()
```

这样修改df_copy时df不会发生变化


```python
df_copy.loc[0, 'gameId'] = 1234
df.loc[0, 'gameId']
print(df_copy.loc[0, 'gameId'])
```

    1234


### 过滤行
实际`df['blueWins'] > 0`返回的是一个true/false的列


```python
df[df['blueWins'] > 0]
# df['blueWins'] > 0
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>...</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>4475365709</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>15201</td>
      <td>7.0</td>
      <td>18060</td>
      <td>221</td>
      <td>59</td>
      <td>-698</td>
      <td>-101</td>
      <td>22.1</td>
      <td>1520.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4493010632</td>
      <td>1</td>
      <td>18</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>14463</td>
      <td>6.4</td>
      <td>15404</td>
      <td>164</td>
      <td>35</td>
      <td>-2411</td>
      <td>-1563</td>
      <td>16.4</td>
      <td>1446.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4509433346</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>16605</td>
      <td>6.8</td>
      <td>18379</td>
      <td>247</td>
      <td>43</td>
      <td>1548</td>
      <td>1574</td>
      <td>24.7</td>
      <td>1660.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4515594785</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>14591</td>
      <td>6.8</td>
      <td>17443</td>
      <td>240</td>
      <td>50</td>
      <td>-3274</td>
      <td>-1659</td>
      <td>24.0</td>
      <td>1459.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4516505202</td>
      <td>1</td>
      <td>15</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16192</td>
      <td>7.0</td>
      <td>18083</td>
      <td>242</td>
      <td>48</td>
      <td>470</td>
      <td>187</td>
      <td>24.2</td>
      <td>1619.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9872</th>
      <td>4527650398</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16399</td>
      <td>7.0</td>
      <td>18001</td>
      <td>216</td>
      <td>58</td>
      <td>-756</td>
      <td>-1</td>
      <td>21.6</td>
      <td>1639.9</td>
    </tr>
    <tr>
      <th>9873</th>
      <td>4527878058</td>
      <td>1</td>
      <td>18</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>6</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>15934</td>
      <td>6.6</td>
      <td>17027</td>
      <td>197</td>
      <td>38</td>
      <td>-2639</td>
      <td>-2364</td>
      <td>19.7</td>
      <td>1593.4</td>
    </tr>
    <tr>
      <th>9874</th>
      <td>4527873286</td>
      <td>1</td>
      <td>17</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>15246</td>
      <td>6.8</td>
      <td>16498</td>
      <td>229</td>
      <td>34</td>
      <td>-2519</td>
      <td>-2469</td>
      <td>22.9</td>
      <td>1524.6</td>
    </tr>
    <tr>
      <th>9875</th>
      <td>4527797466</td>
      <td>1</td>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>15456</td>
      <td>7.0</td>
      <td>18367</td>
      <td>206</td>
      <td>56</td>
      <td>-782</td>
      <td>-888</td>
      <td>20.6</td>
      <td>1545.6</td>
    </tr>
    <tr>
      <th>9878</th>
      <td>4523772935</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>15339</td>
      <td>6.8</td>
      <td>17379</td>
      <td>201</td>
      <td>46</td>
      <td>-927</td>
      <td>58</td>
      <td>20.1</td>
      <td>1533.9</td>
    </tr>
  </tbody>
</table>
<p>4930 rows × 40 columns</p>
</div>



也可以有更复杂的条件


```python
df[(df['blueWardsPlaced'] > 10) & (df['blueWardsPlaced'].isin([9, 15, 17]))]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>...</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>4521474530</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>17285</td>
      <td>6.8</td>
      <td>17254</td>
      <td>203</td>
      <td>28</td>
      <td>1172</td>
      <td>1033</td>
      <td>20.3</td>
      <td>1728.5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4516505202</td>
      <td>1</td>
      <td>15</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16192</td>
      <td>7.0</td>
      <td>18083</td>
      <td>242</td>
      <td>48</td>
      <td>470</td>
      <td>187</td>
      <td>24.2</td>
      <td>1619.2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4482120064</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>17011</td>
      <td>7.2</td>
      <td>18778</td>
      <td>237</td>
      <td>51</td>
      <td>1996</td>
      <td>1804</td>
      <td>23.7</td>
      <td>1701.1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4480384157</td>
      <td>0</td>
      <td>17</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>17027</td>
      <td>7.0</td>
      <td>18129</td>
      <td>231</td>
      <td>60</td>
      <td>1254</td>
      <td>567</td>
      <td>23.1</td>
      <td>1702.7</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4523978853</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>17887</td>
      <td>7.0</td>
      <td>17114</td>
      <td>221</td>
      <td>36</td>
      <td>2472</td>
      <td>1067</td>
      <td>22.1</td>
      <td>1788.7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9845</th>
      <td>4527853173</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>18122</td>
      <td>7.2</td>
      <td>19051</td>
      <td>243</td>
      <td>50</td>
      <td>4676</td>
      <td>3551</td>
      <td>24.3</td>
      <td>1812.2</td>
    </tr>
    <tr>
      <th>9851</th>
      <td>4527637091</td>
      <td>0</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>17300</td>
      <td>7.2</td>
      <td>18342</td>
      <td>236</td>
      <td>55</td>
      <td>2591</td>
      <td>1250</td>
      <td>23.6</td>
      <td>1730.0</td>
    </tr>
    <tr>
      <th>9853</th>
      <td>4527865649</td>
      <td>0</td>
      <td>17</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>17206</td>
      <td>7.0</td>
      <td>18580</td>
      <td>212</td>
      <td>56</td>
      <td>-188</td>
      <td>676</td>
      <td>21.2</td>
      <td>1720.6</td>
    </tr>
    <tr>
      <th>9855</th>
      <td>4527433973</td>
      <td>0</td>
      <td>15</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>16148</td>
      <td>6.6</td>
      <td>17538</td>
      <td>198</td>
      <td>42</td>
      <td>880</td>
      <td>-179</td>
      <td>19.8</td>
      <td>1614.8</td>
    </tr>
    <tr>
      <th>9874</th>
      <td>4527873286</td>
      <td>1</td>
      <td>17</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>15246</td>
      <td>6.8</td>
      <td>16498</td>
      <td>229</td>
      <td>34</td>
      <td>-2519</td>
      <td>-2469</td>
      <td>22.9</td>
      <td>1524.6</td>
    </tr>
  </tbody>
</table>
<p>2205 rows × 40 columns</p>
</div>



### 内置函数
DataFrame内置了一些非常方便的函数，比如算均值mean，最小值min，最大值max等等。 [API文档](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html)


```python
df[cols].mean()
```




    gameId          4.500084e+09
    blueWins        4.990384e-01
    blueAvgLevel    6.916004e+00
    redAvgLevel     6.925316e+00
    dtype: float64



某列Series也是一样有很多内置函数 [API文档](https://pandas.pydata.org/pandas-docs/stable/reference/series.html)


```python
print(df['redAvgLevel'].mean())
print(df['redAvgLevel'].min())
print(df['redAvgLevel'].max())
```

    6.925316327563518
    4.8
    8.2


比如可以快速统计出某一列各个数据出现的次数。


```python
df['blueWins'].value_counts()
```




    0    4949
    1    4930
    Name: blueWins, dtype: int64



### apply函数
apply是个非常强大的函数，可以传入一个函数，这一列中的每个元素都会被这个函数作用，最后返回一个新的列。


```python
df['blueWins'].apply(lambda x: 'win' if x==1 else 'lose')
```




    0       lose
    1       lose
    2       lose
    3       lose
    4       lose
            ... 
    9874     win
    9875     win
    9876    lose
    9877    lose
    9878     win
    Name: blueWins, Length: 9879, dtype: object



也可以不用lambda表达式，定义更复杂的函数


```python
def win(x):
    return 'win' if x == 1 else 'lose'
df_copy['blueWins'] = df['blueWins'].apply(win)
df_copy['blueWins']
```




    0       lose
    1       lose
    2       lose
    3       lose
    4       lose
            ... 
    9874     win
    9875     win
    9876    lose
    9877    lose
    9878     win
    Name: blueWins, Length: 9879, dtype: object



### 列运算


```python
df['brGoldDiff'] = df['blueTotalGold'] - df['redTotalGold']
print(df['brGoldDiff'])
print(df['brGoldDiff'] == df['blueGoldDiff'])
```

    0        643
    1      -2908
    2      -1172
    3      -1321
    4      -1004
            ... 
    9874    2519
    9875     782
    9876   -2416
    9877    -839
    9878     927
    Name: brGoldDiff, Length: 9879, dtype: int64
    0       True
    1       True
    2       True
    3       True
    4       True
            ... 
    9874    True
    9875    True
    9876    True
    9877    True
    9878    True
    Length: 9879, dtype: bool



```python
type(df['brGoldDiff'] == df['blueGoldDiff'])
```




    pandas.core.series.Series



### 小例子
案例1中可能会要求大家对列数据进行离散话，这里举个例子，把某列数据离散化成最小值到最大值的k个区间。

（实际上还有pandas还有cut和qcut函数可以帮助离散化，感兴趣的同学可以进一步了解使用）。


```python
print(df['blueTotalGold'])
def min_max(x, max_v, min_v, k):
    return (x - min_v) // ((max_v - min_v + 1) // k)
min_v = df['blueTotalGold'].min()
max_v = df['blueTotalGold'].max()
df_copy['blueTotalGold'] = df['blueTotalGold'].apply(lambda x: min_max(x, max_v=max_v, min_v=min_v, k=10))
print(df_copy['blueTotalGold'])
df_copy['blueTotalGold'].value_counts()
```

    0       17210
    1       14712
    2       16113
    3       15157
    4       16400
            ...  
    9874    17765
    9875    16238
    9876    15903
    9877    14459
    9878    16266
    Name: blueTotalGold, Length: 9879, dtype: int64
    0       4
    1       3
    2       4
    3       3
    4       4
           ..
    9874    5
    9875    4
    9876    3
    9877    2
    9878    4
    Name: blueTotalGold, Length: 9879, dtype: int64





    4     3236
    3     2753
    5     1970
    2      879
    6      721
    7      198
    1       67
    8       40
    9       12
    0        2
    10       1
    Name: blueTotalGold, dtype: int64



### 其他
其他更多应用和函数可查询[API文档(中文)](https://www.pypandas.cn/docs/) [API文档(英文)](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)


# Numpy
pandas的DataFrame和列Series都可以直接取出数据为numpy的矩阵


```python
print(type(df.values))
print(df.values)
print(type(df['blueTotalGold'].values))
```

    <class 'numpy.ndarray'>
    [[ 4.51915782e+09  0.00000000e+00  2.80000000e+01 ...  1.97000000e+01
       1.65670000e+03  6.43000000e+02]
     [ 4.52337195e+09  0.00000000e+00  1.20000000e+01 ...  2.40000000e+01
       1.76200000e+03 -2.90800000e+03]
     [ 4.52147453e+09  0.00000000e+00  1.50000000e+01 ...  2.03000000e+01
       1.72850000e+03 -1.17200000e+03]
     ...
     [ 4.52771372e+09  0.00000000e+00  2.30000000e+01 ...  2.61000000e+01
       1.83190000e+03 -2.41600000e+03]
     [ 4.52762831e+09  0.00000000e+00  1.40000000e+01 ...  2.47000000e+01
       1.52980000e+03 -8.39000000e+02]
     [ 4.52377294e+09  1.00000000e+00  1.80000000e+01 ...  2.01000000e+01
       1.53390000e+03  9.27000000e+02]]
    <class 'numpy.ndarray'>


数据是没有列名的，类似一个n维数组


```python
blueTotalGold = df['blueTotalGold'].values
print(blueTotalGold)
print(blueTotalGold.dtype)
```

    [17210 14712 16113 ... 15903 14459 16266]
    int64


### 矩阵运算

比n维数组好的地方是支持矩阵运算，比如如果是python数组，要对每个元素+1，需要做循环


```python
python_list = [[1,2,3,4,5], [11,22,33,44,55]]
python_list = [[i + 1 for i in l] for l in python_list]
print(python_list)
```

    [[2, 3, 4, 5, 6], [12, 23, 34, 45, 56]]


但是numpy方便矩阵运算


```python
numpy_array = np.array([[1,2,3,4,5], [11,22,33,44,55]])
print(numpy_array + 1)
```

    [[ 2  3  4  5  6]
     [12 23 34 45 56]]


### 新建特殊矩阵


```python
print(np.zeros((3, 4)))
print(np.ones((2, 2)))
print(np.empty( (2,3) ) )
print(np.arange( 10, 30, 5 ))
print(np.arange(12).reshape(4,3))
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    [[1. 1.]
     [1. 1.]]
    [[8.5029e-320 7.2687e-320 7.9609e-320]
     [7.8571e-320 7.1437e-320 8.0365e-320]]
    [10 15 20 25]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]


### 矩阵属性信息


```python
print(df[cols])
arr = df[cols].values
print(arr)
print(arr.shape)
print(arr.ndim)
print(arr.size)
print(arr.dtype)

```

              gameId  blueWins  blueAvgLevel  redAvgLevel
    0     4519157822         0           6.6          6.8
    1     4523371949         0           6.6          6.8
    2     4521474530         0           6.4          6.8
    3     4524384067         0           7.0          7.0
    4     4436033771         0           7.0          7.0
    ...          ...       ...           ...          ...
    9874  4527873286         1           7.2          6.8
    9875  4527797466         1           7.2          7.0
    9876  4527713716         0           7.0          7.4
    9877  4527628313         0           6.6          7.2
    9878  4523772935         1           7.0          6.8
    
    [9879 rows x 4 columns]
    [[4.51915782e+09 0.00000000e+00 6.60000000e+00 6.80000000e+00]
     [4.52337195e+09 0.00000000e+00 6.60000000e+00 6.80000000e+00]
     [4.52147453e+09 0.00000000e+00 6.40000000e+00 6.80000000e+00]
     ...
     [4.52771372e+09 0.00000000e+00 7.00000000e+00 7.40000000e+00]
     [4.52762831e+09 0.00000000e+00 6.60000000e+00 7.20000000e+00]
     [4.52377294e+09 1.00000000e+00 7.00000000e+00 6.80000000e+00]]
    (9879, 4)
    2
    39516
    float64


### 元素访问
访问第2行


```python
arr[1, :]
```




    array([4.52337195e+09, 0.00000000e+00, 6.60000000e+00, 6.80000000e+00])



访问第3列


```python
arr[:, 2]
```




    array([6.6, 6.6, 6.4, ..., 7. , 6.6, 7. ])



访问前3行，3-最后一列


```python
arr[:3, 2:]
```




    array([[6.6, 6.8],
           [6.6, 6.8],
           [6.4, 6.8]])



### 数学运算


```python
arr[:3, 2:] + 1
```




    array([[7.6, 7.8],
           [7.6, 7.8],
           [7.4, 7.8]])




```python
(arr[:3, 2:] - 1) * 10
```




    array([[56., 58.],
           [56., 58.],
           [54., 58.]])




```python
arr[:3, 2:] ** 2
```




    array([[43.56, 46.24],
           [43.56, 46.24],
           [40.96, 46.24]])




```python
arr[:3, 2:].sum()
```




    40.0




```python
arr[:3, 2:].mean()
```




    6.666666666666667




```python
arr[:3, 2:].max()
arr[:3, 2:].min()
```




    6.4




```python
np.sum(arr[:3, 2:])
np.sin(arr[:3, 2:])
np.cos(arr[:3, 2:])
```




    array([[0.95023259, 0.86939749],
           [0.95023259, 0.86939749],
           [0.99318492, 0.86939749]])



`*`是元素乘，`@`是矩阵乘


```python
A = np.array( [[1,1],
               [0,1]] )
B = np.array( [[2,0],
               [3,4]] )
A * B
```




    array([[2, 0],
           [0, 4]])




```python
A @ B
```




    array([[5, 4],
           [3, 4]])




```python
A + B
```




    array([[3, 1],
           [3, 5]])




```python
A - B
```




    array([[-1,  1],
           [-3, -3]])



按元素相除，除0会抛出warning，返回无穷inf


```python
A / B
```

    /home/shisy13/anaconda3/envs/conda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in true_divide
      """Entry point for launching an IPython kernel.





    array([[0.5 ,  inf],
           [0.  , 0.25]])



数学运算矩阵的大小必须要合法


```python
# A * np.array([1, 2, 3])
A * np.array([1, 2])
```




    array([[1, 2],
           [0, 2]])



### 拷贝
和pandsa一样，一般等号负值是引用，copy之后才是新的对象复制


```python
A_copy = A.copy()
A_copy[0,0] = 2
print(A)
print(A_copy)
```

    [[1 1]
     [0 1]]
    [[2 1]
     [0 1]]


### 小例子
假设要对平均等级做离散化，用类似之前pandas里的实现方式，最大最小值之间以0.2分割区间


```python
blueAvgLevel = df['blueAvgLevel'].values
print(blueAvgLevel)
```

    [6.6 6.6 6.4 ... 7.  6.6 7. ]



```python
print(blueAvgLevel.min(), blueAvgLevel.max())
```

    4.6 8.0


float运算后还是float


```python
(blueAvgLevel - 4.6)/0.1
```




    array([20., 20., 18., ..., 24., 20., 24.])



np.around()可以四舍五入，astype可以变换数据类型


```python
print((blueAvgLevel - 4.6)/0.2)
blueAvgLevel_new = np.around((blueAvgLevel - 4.6)/0.2).astype(int)
print(blueAvgLevel_new)
```

    [10. 10.  9. ... 12. 10. 12.]
    [10 10  9 ... 12 10 12]


可以把array赋值给pandas的某一列


```python
df['blueAvgLevel'] = blueAvgLevel_new
print(df['blueAvgLevel'])
df[cols]
del df
del arr
```

    0       10
    1       10
    2        9
    3       12
    4       12
            ..
    9874    13
    9875    13
    9876    12
    9877    10
    9878    12
    Name: blueAvgLevel, Length: 9879, dtype: int64


### 其他
其他更多应用和函数可查询[API文档(中文)](https://www.numpy.org.cn/reference/) [API文档(英文)](https://numpy.org/doc/stable/)
