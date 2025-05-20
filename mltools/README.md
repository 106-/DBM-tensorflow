mltools
===
機械学習で使ったりする便利プログラムの断片

## 中身
|              |                                                           |
|-------------:|----------------------------------------------------------:|
|       data.py|                  numpy行列からミニバッチを作るためのツール|
|  parameter.py|           パラメータ同士の演算を容易にするための吸収クラス|
|  optimizer.py|                          勾配上昇/降下法のアルゴリズム色々|
|  epochcalc.py|                                             エポック計算機|
|learninglog.py|                                       学習ログの作成を行う|
|     logset.py| learninglog.pyで作成したログファイルの集約やプロットを行う|

## requirements
```
$ pip install -r requirements.txt
```

## logset.py
こんな感じの設定ファイルが必要です.
```
{
    "rcParams": {
        "font.family": "IPAPGothic",
        "font.size": 20
    },
    "subplots_args": {
        "nrows": 1,
        "ncols": 1,
        "figsize": [8, 8]
    },
    "subplots_adjust_args": {
        "wspace": 0.3
    },
    "plots":[
        {
            "title": "学習回数に対するKLDの推移",
            "column": "kl-divergence",
            "xlabel": "epoch",
            "ylabel": "KLD",
            "default_style": {
                "linewidth": 4.0
            },
            "legend_args": {
                "fontsize": 20
            }
        }
    ],
    "data-types":[
        {
            "name":"KLD",
            "typename": "meanfield_montecarlo",
            "filename_includes": "log",
            "style":{
                "linestyle": "dashed"
            }
        }
    ]
}
```
