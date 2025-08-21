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
            nn.Linear(512, 1)
        )
        
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
        return self.classifier(combined).squeeze()

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
            labels = batch['labels'].cpu().numpy()
            pred = model(**inputs).cpu().numpy()
            
            all_preds.extend(pred)
            all_labels.extend(labels)
    
    # 监控预测分布
    print(f"Prediction Stats: Mean={np.mean(all_preds):.2f} Std={np.std(all_preds):.2f}")
    return F.mse_loss(torch.tensor(all_preds), torch.tensor(all_labels)).item(), spearmanr(all_preds, all_labels).correlation

def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # 初始化
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = EnhancedCLIPPairModel("openai/clip-vit-base-patch32").to(device)
    
    # 加载数据
    dataset = load_dataset("glue", "stsb")
    loaders = {
        'train': DataLoader(DynamicSentencePairDataset(tokenizer, dataset['train']), 
                          batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(DynamicSentencePairDataset(tokenizer, dataset['validation']), 
                         batch_size=args.batch_size),
        'test': DataLoader(DynamicSentencePairDataset(tokenizer, dataset['test']), 
                          batch_size=args.batch_size)
    }

    # 数据示例检查
    sample = next(iter(loaders['train']))
    print("\n输入示例检查:")
    print("Input shape:", sample['input_ids1'].shape)
    print("有效token比例:", sample['attention_mask1'].float().mean().item())
    
    # 启动训练
    train(model, loaders, args)

if __name__ == "__main__":
    main()
