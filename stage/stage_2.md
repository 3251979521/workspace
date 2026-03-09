# 2. 编码器与异步操作
## 2.1 编码器可选范围
lip2-so400m-patch16-naflex、siglip-base-patch16-224、siglip2-so400m-patch14-384、siglip2-base-patch16-224、siglip2-giant-opt-patch16-384、siglip2-base-patch16-naflex。
so400m相比于base参数更多，更准确，但是现存消耗大。giant-opt是最精确的，参数最多的。
naflex支持动态分辨率，而不局限于384与224
因此本项目选择的编码器为：siglip2-base-patch16-naflex
## 2.2 demo测试
video1中测试惊喜值，水杯由静止到被手放倒惊喜值有波动但是变化不明显。且均低于阈值0.15
video2中测试惊喜值，从拍水杯由静止到被手放倒，再到场景切换，水杯惊喜值有波动，场景切换波动巨大。
## 2.3 demo测试v1改进
我们发现由于画面中变化的patch其实较少，代码中对所有patch进行了平均化处理，所以导致画面整体变化被中和，我们改进，保留模型输出的每一个图像块（Patch）的特征，让前后两张图对应位置的 Patch 一一比对。最后取变化最大的前 5% 的 patch 的平均值来计算