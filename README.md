DBM-tensorflow
===
Implementation of deep Boltzmann machine using TensorFlow.

[numpyでの実装](https://github.com/106-/Deep-Boltzmann-Machine)を元にTensorFlowで実装し直したものです.
GPUが使えたり, numpyのボトルネック(計算過程がいちいちメモリに配置される)が無い分こっちのが早いです.
使い方もほぼ同じ.

こちらのプロジェクトは学習アルゴリズムの実験用の実装となっています.以下の3つのアルゴリズムにて学習を行うことができます。

- 平均場近似&モンテカルロ積分を利用した, Salakhutdinov & Hintonによる学習方法
- 空間モンテカルロ積分(Spatial Monte Carlo Integration; S-MCI)を利用した, M. Yasudaによる学習方法
- 厳密値による学習

numpyでの実装では処理速度的な限界から3層までの実装となっていますが, この実装では好きなだけ層を重ねることができます(2-SMCI法を除く).

## 使い方
_Python3.12が必要です._

[uv](https://docs.astral.sh/uv/) を使って依存関係を管理しています。

```
$ git clone https://github.com/106-/DBM-tensorflow.git DBM
$ cd DBM
```
依存関係のインストール
```
$ make sync
```
あとは実験条件をjsonファイルに記述し実行すればOK
```
$ make train                                                   # デフォルト設定で実行
$ make train CONFIG=./config/3-layer/exact/double.json EPOCH=100
```

各 `config/` サブディレクトリは以下のアルゴリズムに対応しています.

| config ディレクトリ    | データ期待値     | モデル期待値     |
|------------------------|------------------|------------------|
| `exact`                | 厳密値           | 厳密値           |
| `meanfield_montecarlo` | 平均場近似       | モンテカルロ     |
| `1-smci`               | 1次SMCI          | 1次SMCI          |
| `2-smci`               | 2次SMCI          | 2次SMCI          |
`result/` 下に実験の結果が出力されます. `*_log.json` のものがKLDと対数尤度のログ, `*_model.json` がモデルの重みパラメータのファイルです.

# References
- [1]: R. Salakhutdinov and G. Hinton: [Deep Boltzmann Machines](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf), Artificial intelligence and statistics, pp.448-455, 2009.
- [2]: M. Yasuda: [Monte Carlo Integration Using Spatial Structure of Markov Random Field](https://journals.jps.jp/doi/10.7566/JPSJ.84.034001), Journal of the Physical Society of Japan 84(3), 2015.
- [3]: M. Yasuda: [Learning Algorithm of Boltzmann Machine Based on Spatial Monte Carlo Integration Method](https://www.mdpi.com/1999-4893/11/4/42/htm),  Algorithms 11(4), 2018.
- [4]: M. Yasuda & K. Uchizawa: [A Generalization of Spatial Monte Carlo Integration](https://arxiv.org/abs/2009.02165), arXiv:2009.02165, 2020
