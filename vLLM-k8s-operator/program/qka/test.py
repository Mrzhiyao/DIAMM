
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.stats import spearmanr

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class DynamicSentencePairDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length_percentile=95):
        self.tokenizer = tokenizer
        
        # 自动计算最佳max_length
        lengths = []
        for s1, s2 in zip(dataset['sentence1'], dataset['sentence2']):
            lengths.append(len(tokenizer.tokenize(s1)))
            lengths.append(len(tokenizer.tokenize(s2)))
        self.max_length = int(np.percentile(lengths, max_length_percentile)) + 5  # 增加缓冲
        print(f"Automatically set max_length to {self.max_length}")
        
        self.sentence_pairs = list(zip(dataset['sentence1'], dataset['sentence2']))
        self.labels = dataset['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]

        # 动态填充编码
        enc1 = self.tokenizer(
            sent1, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        enc2 = self.tokenizer(
            sent2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids1': enc1['input_ids'].squeeze(),
            'attention_mask1': enc1['attention_mask'].squeeze(),
            'input_ids2': enc2['input_ids'].squeeze(),
            'attention_mask2': enc2['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
from nlpaug.augmenter.word import ContextualWordEmbsAug

class AugmentedDataset(DynamicSentencePairDataset):
    def __init__(self, tokenizer, dataset, aug_prob=0.3):
        super().__init__(tokenizer, dataset)
        self.aug = ContextualWordEmbsAug(model_path='bert-base-uncased', device='cuda', aug_p=0.3)
        self.aug_prob = aug_prob
        
    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]
        
        # 概率性增强
        if torch.rand(1) < self.aug_prob:
            sent1 = self.aug.augment(sent1)
        if torch.rand(1) < self.aug_prob:
            sent2 = self.aug.augment(sent2)
        
        # 其余处理同父类
        return super().__getitem__(idx)
    
class EnhancedCLIPPairModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # 解冻最后3层 + 分类头
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for i in [-2, -1]:
            for param in self.text_encoder.text_model.encoder.layers[i].parameters():
                param.requires_grad = True

        # 增强型特征提取
        hidden_size = self.text_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*3, 512),  # 新增差异特征
            nn.ReLU(),
            nn.Dropout(0.5),  # 提升Dropout比例
            nn.LayerNorm(512), # 添加层标准化
            nn.Linear(512, 1),
            nn.Sigmoid()  # 先压缩到[0,1]
        )
        
        self.scale = nn.Parameter(torch.tensor(1.5, requires_grad=True))

    def _masked_pooling(self, hidden_states, attention_mask):
        """基于注意力掩码的加权平均池化"""
        weights = attention_mask.unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # 编码句子特征
        out1 = self.text_encoder(input_ids1, attention_mask1).last_hidden_state
        out2 = self.text_encoder(input_ids2, attention_mask2).last_hidden_state
        
        # 增强特征提取
        emb1 = self._masked_pooling(out1, attention_mask1)
        emb2 = self._masked_pooling(out2, attention_mask2)
        diff = torch.abs(emb1 - emb2)  # 差异特征
        
        # 组合特征
        combined = torch.cat([emb1, emb2, diff], dim=1)
        # return self.classifier(combined).squeeze()
        output = self.classifier(combined).squeeze()
        return output * self.scale  # 最终输出范围[0,1.5]
    
def train(model, loaders, args):
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-3},
        {'params': model.text_encoder.parameters(), 'lr': 1e-5}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    best_corr = -1

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in tqdm(loaders['train'], desc="Training"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            # print(labels)
            optimizer.zero_grad()
            pred = model(**inputs)
            loss = F.mse_loss(pred, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # 验证阶段
        val_loss, val_corr = evaluate(model, loaders['val'])
        scheduler.step(val_loss)
        print(f"Train Loss: {train_loss/len(loaders['train']):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Spearman: {val_corr:.4f}")
        
        # 保存最佳模型
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), "best_model.pth")
    
    # 最终测试
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_corr = evaluate(model, loaders['test'])
    print(f"\nFinal Test: Loss={test_loss:.4f} | Spearman={test_corr:.4f}")

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].cpu().numpy().flatten()
            pred = model(**inputs).cpu().numpy().flatten()
            all_preds.extend(pred.tolist())
            all_labels.extend(labels.tolist())
    
    # 转换为PyTorch张量
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    
    # 使用PyTorch函数计算统计量
    print(f"Prediction Stats: Mean={all_preds.mean().item():.2f} Std={all_preds.std().item():.2f}")
    
    return F.mse_loss(all_preds, all_labels).item(), spearmanr(all_preds.numpy(), all_labels.numpy()).correlation


def test_model(model, loader, device, sample_num=50):
    model.eval()
    all_preds = []
    all_labels = []
    sample_data = []  # 用于保存示例数据
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].cpu().numpy().flatten()
            preds = model(**inputs).cpu().numpy().flatten()
            
            # 保存第一批的部分样本用于展示
            if len(sample_data) < sample_num:
                for i in range(min(sample_num, len(labels))):
                    sample_data.append((
                        preds[i],
                        labels[i],
                        loader.dataset.sentence_pairs[i][0],  # 原始句子1
                        loader.dataset.sentence_pairs[i][1]   # 原始句子2
                    ))
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    # 转换为张量
    preds_tensor = torch.tensor(all_preds)
    labels_tensor = torch.tensor(all_labels)
    
    # 计算各项指标
    mse = F.mse_loss(preds_tensor, labels_tensor)
    mae = F.l1_loss(preds_tensor, labels_tensor)
    spearman = spearmanr(all_preds, all_labels).correlation
    pearson = np.corrcoef(all_preds, all_labels)[0, 1]
    
    # 打印综合结果
    print("\n" + "="*60)
    print(f"{'Test Results':^60}")
    print("="*60)
    print(f"MSE Loss: \t{mse.item():.4f}")
    print(f"MAE Loss: \t{mae.item():.4f}")
    print(f"Spearman Correlation: \t{spearman:.4f}")
    print(f"Pearson Correlation: \t{pearson:.4f}")
    print(f"Prediction Range: [{preds_tensor.min().item():.2f}, {preds_tensor.max().item():.2f}]")
    print(f"True Value Range: [{labels_tensor.min().item():.2f}, {labels_tensor.max().item():.2f}]")
    
    # 显示示例预测
    print("\nExample Predictions:")
    for i, (pred, label, s1, s2) in enumerate(sample_data[:sample_num]):
        print(f"\nSample {i+1}:")
        print(f"Sentence 1: {s1}")
        print(f"Sentence 2: {s2}")
        print(f"Predicted: {pred:.2f} | True: {label:.2f} | Diff: {abs(pred - label):.2f}")
    
    # 寻找最大误差样本
    errors = torch.abs(preds_tensor - labels_tensor)
    worst_idx = errors.argmax().item()
    print("\nWorst Prediction:")
    print(f"Sentence 1: {loader.dataset.sentence_pairs[worst_idx][0]}")
    print(f"Sentence 2: {loader.dataset.sentence_pairs[worst_idx][1]}")
    print(f"Predicted: {preds_tensor[worst_idx].item():.2f} | True: {labels_tensor[worst_idx].item():.2f}")
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'spearman': spearman,
        'pearson': pearson
    }




def test_only():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = EnhancedCLIPPairModel("openai/clip-vit-base-patch32").to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    
    # 1. 正确加载所有数据文件
    file_paths = [
        "E:/com_work/combined/combined_data.jsonl",
        "E:/com_work/saved_data_chunks/part-0000.jsonl",
        "E:/com_work/controlled_data/formatted_data.jsonl",
        "E:/com_work/balanced_data/data_part_0000.jsonl"
    ]

    # 正确加载方式：使用列表形式加载所有文件
    dataset = load_dataset("json", data_files=file_paths, split="train")

    # 2. 使用dataset自带的划分方法（更高效）
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    temp_test = split_dataset["test"].train_test_split(test_size=0.3, seed=42)

    # 重组最终数据集
    final_dataset = {
        "train": split_dataset["train"],
        "validation": temp_test["train"],
        "test": temp_test["test"]
    }
    loaders = {
        'train': DataLoader(DynamicSentencePairDataset(tokenizer, final_dataset['train']), 
                          batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(DynamicSentencePairDataset(tokenizer, final_dataset['validation']), 
                         batch_size=args.batch_size),
        'test': DataLoader(DynamicSentencePairDataset(tokenizer, final_dataset['test']), 
                          batch_size=args.batch_size)
    }
    test_loader = DataLoader(
        DynamicSentencePairDataset(tokenizer, temp_test["test"]),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_model(model, test_loader, device)


def static_predict(model, tokenizer, device, sentence1, sentence2):
    """静态预测两个句子的相似度"""
    model.eval()
    
    # 预处理与训练时一致
    inputs = tokenizer(
        [sentence1, sentence2],
        max_length=model.text_encoder.config.max_position_embeddings,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 转换为模型输入格式
    input_ids1 = inputs['input_ids'][0].unsqueeze(0).to(device)
    attention_mask1 = inputs['attention_mask'][0].unsqueeze(0).to(device)
    input_ids2 = inputs['input_ids'][1].unsqueeze(0).to(device)
    attention_mask2 = inputs['attention_mask'][1].unsqueeze(0).to(device)
    
    with torch.no_grad():
        similarity = model(
            input_ids1=input_ids1,
            attention_mask1=attention_mask1,
            input_ids2=input_ids2,
            attention_mask2=attention_mask2
        ).item()
    
    # 转换为1.5分制
    scaled_score = similarity * (1.5 / model.scale.item())
    return {
        'sentence1': sentence1,
        'sentence2': sentence2,
        'raw_score': similarity,
        'scaled_score': scaled_score
    }


if __name__ == "__main__":
    # test_only()
    # 初始化模型和tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = EnhancedCLIPPairModel("openai/clip-vit-base-patch32").to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    
    # 预定义测试用例
    test_cases = [
        ("State schools (also known as public schools or government schools) generally refer to primary or secondary schools mandated for or offered to all children without charge paid for, in whole or in part, by taxation .", "The term may also refer to institutions of post-secondary education funded, in whole or in part, and overseen by government."),
        ("A partly submerged glacier cave on Perito Moreno Glacier .", "The ice facade is approximately 60 m high"),
        ("The quick brown fox jumps", "The agile brown fox switch")
    ]
    
    # 批量测试
    for s1, s2 in test_cases:
        result = static_predict(model, tokenizer, device, s1, s2)
        print(f"\n句子1: {result['sentence1']}")
        print(f"句子2: {result['sentence2']}")
        print(f"原始分数: {result['raw_score']:.4f}")
        print(f"标准分数: {result['scaled_score']:.2f}/1.5")
        print("="*60)
