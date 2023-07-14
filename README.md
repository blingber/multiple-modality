# multiple-modality

## 项目概述
本次实验利用BERT模型作为文本特征提取器，使用了预训练的ResNet50模型作为图像特征提取器，Model将文本和图像特征进行融合，并使用全连接层进行情感分类。
- 给定配对的文本和图像，预测对应的情感标签。
- 三分类任务：positive, neutral, negative。

## 库
1. torch: 本实验使用PyTorch深度学习框架.
2. os: 提供操作系统相关的功能，如路径操作和文件读写。
3. json: 用于处理JSON数据。
4. numpy: 用于处理数值计算和数组操作。
5. PIL: 用于图像处理和操作。
6. argparse: 用于解析命令行参数。
7. chardet: 用于自动检测文本的编码格式。
8. warnings: 用于忽略警告信息。
9. logging: 用于设置日志输出级别。
10. train_test_split: 用于将数据集划分为训练集和验证集。
11. AdamW: 提供AdamW优化器。
12. tqdm: 用于显示进度条。
13. nn: 提供神经网络相关的功能和模块。
14. AutoModel: 提供预训练的BERT模型。
15. resnet50: 提供预训练的ResNet-50模型。
16. DataLoader: 用于加载训练、验证和测试数据的数据加载器。
17. Dataset: 提供自定义数据集的基类。
18. pad_sequence: 用于对序列进行填充。
19. AutoTokenizer: 提供自动选择和加载预训练的分词器。
20. transforms: 提供图像预处理和转换。
21. accuracy_score: 用于计算准确率指标。

## 安装指南

```
pip install -r requirements.txt
conda install --yes --file requirements.txt
```
仓库中有requirements.txt记录了当前程序的所有依赖包及其精确版本号，使用上述命令其中之一可配置相同环境。

## 演示
在main.py根目录下输入下述命令（在GPU下10个epoch大概需要12分钟，需要耐心等待）

text与img训练
`python main.py --text --img --train --epoch 10`·

消融
`python main.py --text --train --epoch 10`
`python main.py --img --train --epoch 10`

预测
`python main.py --model_path ./output/mymodel.bin --test`

## 文件结构
```
| |---train.json
| |---test.json
| |--text
| |--img
|---data
|---train.txt
|---test_without_label
|---main.py
```

## 实验结果
```
多模态融合模型在验证集上的结果：0.68
```
```
消融实验结果:
text：0.62
img：0.58
```

