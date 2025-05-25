DBM-tensorflow
===
Implementation of deep Boltzmann machine using TensorFlow.

[numpyでの実装](https://github.com/106-/Deep-Boltzmann-Machine)を元にTensorFlowで実装し直したものです.
GPUが使えたり, numpyのボトルネック(計算過程がいちいちメモリに配置される)が無い分こっちのが早いです.
使い方もほぼ同じ.

numpyでの実装では処理速度的な限界から3層までの実装となっていますが, この実装では好きなだけ層を重ねることができます(2-SMCI法を除く).

## 使い方
_Python3.12が必要です._

tensorflowという大きめのモジュールを使用するため、`python -m venv .venv` などで環境を分離するのをおすすめします。

```
$ git clone https://github.com/106-/DBM-tensorflow.git DBM
$ cd DBM
```
サブリポジトリのファイルを持ってくる
```
$ git submodule update --init --recursive
```
必要モジュールのインストール
```
$ pip install -r ./requirements.txt
```
あとは実験条件をjsonファイルに記述し実行すればOK
```
./train_main.py ./config/3-layer/exact/double.json 100     # 100は実行エポック
```
`result/` 下に実験の結果が出力されます. `*_log.json` のものがKLDと対数尤度のログ, `*_model.json` がモデルの重みパラメータのファイルです.

# References
- [1]: R. Salakhutdinov and G. Hinton: [Deep Boltzmann Machines](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf), Artificial intelligence and statistics, pp.448-455, 2009.
- [2]: M. Yasuda: [Monte Carlo Integration Using Spatial Structure of Markov Random Field](https://journals.jps.jp/doi/10.7566/JPSJ.84.034001), Journal of the Physical Society of Japan 84(3), 2015.
- [3]: M. Yasuda: [Learning Algorithm of Boltzmann Machine Based on Spatial Monte Carlo Integration Method](https://www.mdpi.com/1999-4893/11/4/42/htm),  Algorithms 11(4), 2018.
- [4]: M. Yasuda & K. Uchizawa: [A Generalization of Spatial Monte Carlo Integration](https://arxiv.org/abs/2009.02165), arXiv:2009.02165, 2020