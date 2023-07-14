import os
import torch
import argparse
import json
import chardet
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torchvision import transforms
from sklearn.metrics import accuracy_score

def formating(path, dir, out_path):
    data = []  # 存储格式化后的数据
    with open(path) as f:
        for line in f.readlines():  
            guid, label = line.replace('\n', '').split(',')  # 将每行数据按逗号分隔成guid和label
            text_path = os.path.join(dir, (guid + '.txt'))  
            if guid == 'guid':
                continue  # 如果guid为'guid'，跳过标题行
            with open(text_path, 'rb') as texts:  
                thetext = texts.read()  # 读取文本内容
                encode = chardet.detect(thetext)  # 检测文本的编码格式
                try:
                    text = thetext.decode(encode['encoding'])  
                except:
                    try:
                        text = thetext.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')  #
                    except:
                        print('not is0-8859-1', guid)  # 如果无法解码，则输出错误信息并跳过该行数据
                        continue
            text = text.strip().strip('\n').strip(' ').strip('\r')  
            data.append({'guid': guid, 'label': label, 'text': text})  # 将处理后的数据添加到列表中
    with open(out_path, 'w') as ff:  # 打开输出文件
        json.dump(data, ff)  # 将数据以JSON格式写入文件

# 将数据返回元组列表
def listing(path):
    data = []
    with open(path) as f:
        json_file = json.load(f)
        for d in json_file:
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid': 
               continue
            img_path = os.path.join(data_dir, (guid + '.jpg'))
            img = Image.open(img_path)
            img.load()
            data.append((guid, text, img, label))
        f.close()
    return data

class DataOperating(Dataset):
    def __init__(self, guids, texts, imgs, labels):
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return len(self.guids)
    def __getitem__(self, index):   
        #根据index获得详细数据
        return self.guids[index], self.texts[index],self.imgs[index], self.labels[index]
    def collate_fn(self, batch):
        guids = [sample[0] for sample in batch]
        texts = [torch.LongTensor(sample[1]) for sample in batch]
        imgs = torch.FloatTensor([np.array(sample[2]).tolist() for sample in batch])
        labels = torch.LongTensor([sample[3] for sample in batch])
        # 创建文本掩码
        mask = [torch.ones_like(text) for text in texts]
        # 填充文本序列
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        # 填充文本掩码
        padded_mask = pad_sequence(mask, batch_first=True, padding_value=0).gt(0)
        return guids, padded_texts, padded_mask, imgs, labels

class Voc_label:
    def __init__(self):
        self.label_1 = {}
        self.label_2 = {}
    def __len__(self):
        return len(self.label_1)
    def add_label(self, label):
        if label not in self.label_1:
            self.label_1.update({label: len(self.label_1)})
            self.label_2.update({len(self.label_2): label})
    def label_to_id(self, label):
        return self.label_1.get(label)
    def id_to_label(self, id):
        return self.label_2.get(id)

class Processor:
    def __init__(self):
        self.voc_label = Voc_label()  # 创建一个词汇标签对象，用于处理标签相关操作 
    def __call__(self, data, params):
        return self.to_loader(data, params)  # 调用 to_loader() 方法将数据转换为 DataLoader
    def encode(self, data):
        voc_label = self.voc_label
        voc_label.add_label('positive')  
        voc_label.add_label('neutral')  
        voc_label.add_label('negative')  
        voc_label.add_label('null')  # 添加标签 'null' 到词汇标签中
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # 创建一个文本 tokenizer 对象
        def get_resize(image_size):
            for i in range(20):
                if 2 ** i >= image_size:
                    return 2 ** i
            return image_size   
        img_transform = transforms.Compose([
            transforms.Resize(get_resize(224)),  # 调整图像大小为指定尺寸
            transforms.CenterCrop(224),  # 将图像中心进行裁剪
            transforms.RandomHorizontalFlip(0.5),  # 以0.5的概率对图像进行水平翻转
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像张量
            ]) 
        guids = []  # 存储处理后的guid
        e_texts = []  # 存储处理后的文本
        e_imgs = []  # 存储处理后的图像
        e_labels = []  # 存储处理后的标签    
        for line in data:
            guid, text, img, label = line
            guids.append(guid)
            text.replace('#', '') 
            tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')  # 将文本转换为tokens
            e_texts.append(tokenizer.convert_tokens_to_ids(tokens))  # 将tokens转换为文本的编码
            e_imgs.append(img_transform(img))  # 对图像进行预处理和转换
            e_labels.append(voc_label.label_to_id(label))  # 将标签转换为标签的编码   
        return guids, e_texts, e_imgs, e_labels
    def decode(self, outputs):
        voc_label = self.voc_label
        formated_outputs = ['guid,tag']  # 存储解析后的输出  
        for guid, label in tqdm(outputs, desc='解析'):
            formated_outputs.append((str(guid) + ',' + voc_label.id_to_label(label)))  # 将解析后的输出添加到列表中
        return formated_outputs
    def metric(self, inputs, outputs):
        return accuracy_score(inputs, outputs)  # 计算准确率指标
    def to_dataset(self, data):
        dataset_inputs = self.encode(data)  # 对数据进行编码处理
        result = DataOperating(*dataset_inputs)  # 创建 Dataset 对象
        return result
    def to_loader(self, data, params):
        dataset = self.to_dataset(data)  # 将数据转换为 Dataset 对象
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)  # 创建 DataLoader 并返回

class TextModel(nn.Module): #文本模型,包含了BERT模型和一个线性转换层,得到最终的文本特征表示
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.hidden_transform = nn.Sequential(nn.Dropout(0.2),nn.Linear(self.bert.config.hidden_size, 64),nn.ReLU(inplace=True))################
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 输入通过BERT模型进行编码
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = bert_output['pooler_output']  # 获取池化后的表示
        # 对池化后的表示进行线性变换和激活函数处理
        transformed_output = self.hidden_transform(pooled_output)
        return transformed_output

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        # 冻结 ResNet50 模型的参数
        for param in self.full_resnet.parameters():
            param.requires_grad = False   
        # 删除最后的全连接层，将最后一个卷积层的输出展平为一维向量
        self.resnet = nn.Sequential(*[layer for layer in self.full_resnet.children()][:-1])
        self.flatten = nn.Flatten()   
        # 添加线性变换层
        self.trans = nn.Sequential(nn.Dropout(0.2),nn.Linear(self.full_resnet.fc.in_features, 64),nn.ReLU(inplace=True))
    def forward(self, imgs):
        features = self.resnet(imgs)
        flattened = self.flatten(features)
        return self.trans(flattened)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.text_model = TextModel()  # 创建文本模型对象
        self.img_model = ImageModel()  # 创建图像模型对象   
        # 全连接分类器
        self.classifier = nn.ModuleList([
            nn.Sequential(nn.Dropout(0.5), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 3),
                          nn.Softmax(dim=1)),
            nn.Sequential(nn.Dropout(0.5), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 3),
                          nn.Softmax(dim=1))
        ])
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.71, 9.5, 3.33]))  # 定义损失函数
    def forward(self, texts, mask, imgs, labels=None):
        text_feature = self.text_model(texts, mask)  # 文本模型的前向传播，得到文本特征表示
        img_feature = self.img_model(imgs)  # 图像模型的前向传播，得到图像特征表示 
        text_vec = self.classifier[0](text_feature)  # 文本特征经过全连接分类器得到预测结果
        img_vec = self.classifier[1](img_feature)  # 图像特征经过全连接分类器得到预测结果    
        prob_vec = torch.softmax((text_vec + img_vec), dim=1)  # 将文本和图像的预测结果进行加和并进行 softmax 归一化
        p_labels = torch.argmax(prob_vec, dim=1)  # 预测的标签
        if labels is None:
            return p_labels  # 如果没有提供真实标签，则只返回预测标签
        else:
            loss = self.loss_func(prob_vec, labels)  # 计算损失函数
            return p_labels, loss  # 返回预测标签和损失

class Trainer():
    def __init__(self, processor, model, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.processor = processor
        self.model = model.to(device)
        self.device = device
        bert_params = set(self.model.text_model.bert.parameters())
        resnet_params = set(self.model.img_model.full_resnet.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)
        no_decay = ['bias', 'LayerNorm.weight'] # 不进行权重衰减的参数列表
        params = [{
            'params': [param for name, param in self.model.text_model.bert.named_parameters() if not any(nd in name for nd in no_decay)],
            'lr': 1e-6,
            'weight_decay': 0.0
        },
        {
            'params': [param for name, param in self.model.text_model.bert.named_parameters() if any(nd in name for nd in no_decay)],
            'lr': 1e-6,
            'weight_decay': 0.0
        },
        {
            'params': [param for name, param in self.model.img_model.full_resnet.named_parameters() if not any(nd in name for nd in no_decay)],
            'lr': 1e-6,
            'weight_decay': 0.0
        },
        {
            'params': [param for name, param in self.model.img_model.full_resnet.named_parameters() if any(nd in name for nd in no_decay)],
            'lr': 1e-6,
            'weight_decay': 0.0
        },
        {
            'params': other_params,
            'lr': 1e-6,
            'weight_decay': 0.0
        },]
        self.optimizer = AdamW(params, lr=3e-5)

    def training(self, train_loader):
        """
        参数:
            train_loader (DataLoader): 训练数据的数据加载器
        返回:
            train_loss (float): 训练损失
            loss_list (list): 每个批次的损失列表
        """
        self.model.train()  # 设置模型为训练模式
        loss_list = []  # 每个批次的损失列表
        t_labels, p_labels = [], []  # 真实标签和预测标签列表
        for batch in tqdm(train_loader, desc='训练'):
            guids, texts, mask, imgs, labels = batch
            texts, mask, imgs, labels = texts.to(self.device), mask.to(self.device), imgs.to(self.device), labels.to(self.device)
            pred, loss = self.model(texts, mask, imgs, labels=labels)  # 前向传播计算预测值和损失
            loss_list.append(loss.item())  # 记录当前批次的损失值
            t_labels.extend(labels.tolist())  # 记录真实标签
            p_labels.extend(pred.tolist())  # 记录预测标签
            self.optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播计算梯度
            self.optimizer.step()  # 更新模型参数
        train_loss = round(sum(loss_list)/len(loss_list), 4)  # 计算平均训练损失
        return train_loss, loss_list


    def valid(self, val_loader):
        """
        参数:
            val_loader (DataLoader): 验证数据的数据加载器
        返回:
            val_loss (float): 验证损失
            metrics: (float): 验证指标值（如准确率）
        """
        self.model.eval()  # 设置模型为评估模式
        t_labels = []  # 真实标签列表
        p_labels = []  # 预测标签列表
        val_loss = 0  # 验证损失
        for batch in tqdm(val_loader, desc='验证'):
            guids, texts, mask, imgs, labels = batch
            texts, mask, imgs, labels = texts.to(self.device), mask.to(self.device), imgs.to(self.device), labels.to(self.device)
            pred, loss = self.model(texts, mask, imgs, labels=labels)  # 前向传播计算预测值和损失
            val_loss += loss.item()  # 累计验证损失
            t_labels.extend(labels.tolist())  # 记录真实标签
            p_labels.extend(pred.tolist())  # 记录预测标签
        metrics = self.processor.metric(t_labels, p_labels)  # 计算验证指标（如准确率）
        return val_loss/len(val_loader), metrics


    def predict(self, test_loader):
        """
        参数:
            test_loader (DataLoader): 测试数据的数据加载器
        返回:
            predictions (list): 预测结果列表，包含(guid, label)元组
        """
        self.model.eval()  # 设置模型为评估模式
        p_guids = []  # 预测guid列表
        p_labels = []  # 预测标签列表
        for batch in tqdm(test_loader, desc='预测'):
            guids, texts, mask, imgs, labels = batch
            texts, mask, imgs = texts.to(self.device), mask.to(self.device), imgs.to(self.device)
            pred = self.model(texts, mask, imgs)  # 前向传播计算预测值
            p_guids.extend(guids)  # 记录预测guid
            p_labels.extend(pred.tolist())  # 记录预测标签
        predictions = [(guid, label) for guid, label in zip(p_guids, p_labels)]  # 构建预测结果列表
        return predictions


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--epoch', default=10,type=int)
parser.add_argument('--text_pretrained_model', default='bert-base-uncased', type=str)
parser.add_argument('--model_path', default=None, type=str)
args = parser.parse_args()

weight_decay = 1e-2 #衰减系数
learning_rate = args.lr
epoch = args.epoch
bert_name = args.text_pretrained_model
model_path = args.model_path

root_path = root_path = os.getcwd()
data_dir = os.path.join(root_path, './data')
train_path = os.path.join(root_path, 'data/train.json')
test_path = os.path.join(root_path, 'data/test.json')
output_path = os.path.join(root_path, 'output')
output_test_path = os.path.join(output_path, 'test.txt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

processor = Processor()
model = MyModel()
trainer = Trainer(processor, model, device)

checkout_params = {'batch_size': 4, 'shuffle': False}# 检查数据集参数字典，用于加载检查数据集，每批次大小为4，不进行数据洗牌
train_params = {'batch_size': 16}# 训练数据集参数字典，用于加载训练数据集，每批次大小为16，进行数据洗牌
val_params = {'batch_size': 16, 'shuffle': False}
test_params = {'batch_size': 8, 'shuffle': False}

def train():
    formating(os.path.join(root_path, 'train.txt'),
                os.path.join(root_path, './data'), os.path.join(root_path, './data/train.json'))
    data = listing(train_path)
    train_data, val_data = train_test_split(data, train_size=(0.8), test_size=0.2)
    train_loader = processor(train_data, train_params)
    val_loader = processor(val_data, val_params)
    best_acc = 0
    for i in range(epoch):
        print('Epoch '+ str(i+1))
        t_loss, t_loss_list = trainer.training(train_loader)
        print('训练时损失: {}'.format(t_loss))
        v_loss, v_acc = trainer.valid(val_loader)
        print('验证集损失: {}'.format(v_loss))
        print('验证集准确度: {}'.format(v_acc))
        if v_acc > best_acc:#若准确度更高，保留新的模型
            save = model.module if hasattr(model, 'module') else model 
            output_model = os.path.join(output_path, "mymodel.bin")
            torch.save(save.state_dict(), output_model)
            best_acc = v_acc

def test():
    formating(os.path.join(root_path, 'test_without_label.txt'),os.path.join(root_path, './data'), os.path.join(root_path, './data/test.json'))
    test_data = listing(test_path)
    test_loader = processor(test_data, test_params)   
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    outputs = trainer.predict(test_loader)
    outputs_last = processor.decode(outputs)
    with open(output_test_path, 'w') as f:
        for line in tqdm(outputs_last):
            f.write(line)
            f.write('\n')
        f.close()

if __name__ == "__main__":
    if args.train:
        train()
    if args.test:
        test()