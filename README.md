# 1. Outline
记录-解读 并整理 我所收集到好的code exmaple, 建立自己的代码库非常重要
### 1. 简单整理写小心得，有规模了再整合
1. 诸如 torch.device 确实可以打包写到 constant.py (contant.py 同时收集 文件的传参以及 sys.args 的传参)，然后啥文件要，就 constant.xxx 传给他。这样就避免了 其他文件大量的从各处接受参数的 难看code还难改。。。(现在只需要 constant.py 干完全部这样的活包揽全部的实现细节，然后其他文件 import consant, 想用啥参数再去 具体获得即可。这才是真的解耦代码，高效率写代码！！！)
