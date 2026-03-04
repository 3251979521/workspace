# 2. 编码器与异步操作
## 2.1 编码器可选范围
lip2-so400m-patch16-naflex、siglip-base-patch16-224、siglip2-so400m-patch14-384、siglip2-base-patch16-224、siglip2-giant-opt-patch16-384、siglip2-base-patch16-naflex。
so400m相比于base参数更多，更准确，但是现存消耗大。giant-opt是最精确的，参数最多的。
naflex支持动态分辨率，而不局限于384与224
## 2.2 demo测试
video1中测试惊喜值，水杯由静止到被手放倒惊喜值有波动但是变化不明显。且均低于阈值0.15
video2中测试惊喜值，从拍水杯由静止到被手放倒，再到场景切换，水杯惊喜值有波动，场景切换波动巨大。