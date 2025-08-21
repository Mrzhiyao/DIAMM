import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.stats import spearmanr
import csv
import time

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集类
class SentencePairDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length=77):
        self.tokenizer = tokenizer
        self.sentence_pairs = list(zip(dataset['sentence1'], dataset['sentence2']))
        self.labels = dataset['label']
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]
        
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

# 改进的模型定义（返回嵌入）
class CLIPPairModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.clip = CLIPTextModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.hidden_size*2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        self.temperature = nn.Parameter(torch.tensor(0.1))  # 可学习温度参数

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # 获取两个句子的嵌入
        # sentence1 = tokenizer.decode(batch['input_ids1'][0].cpu().numpy())
        # sentence2 = tokenizer.decode(batch['input_ids2'][0].cpu().numpy())

        out1 = self.clip(input_ids=input_ids1, attention_mask=attention_mask1)
        out2 = self.clip(input_ids=input_ids2, attention_mask=attention_mask2)
        
        emb1 = out1.last_hidden_state[:, 0, :]  # [CLS]向量
        emb2 = out2.last_hidden_state[:, 0, :]
        
        # 回归预测
        combined = torch.cat([emb1, emb2], dim=1)
        # print('combined', combined.shape)
        pred = self.classifier(combined).squeeze()
        
        return pred, emb1, emb2

# 组合损失函数
def hybrid_loss(pred, labels, emb1, emb2, temperature):
    # MSE回归损失
    mse_loss = F.mse_loss(pred, labels)
    
    # 对比学习损失
    batch_size = emb1.size(0)
    
    # 归一化嵌入
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    # print(emb1.shape, emb2.shape,'shape')
    # 相似度矩阵
    sim_matrix = torch.matmul(emb1, emb2.T) / temperature
    
    # 正样本掩码（对角线）
    pos_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    
    # 对比损失计算
    exp_sim = torch.exp(sim_matrix)
    pos_samples = exp_sim[pos_mask]
    neg_samples = exp_sim.sum(dim=1) - pos_samples
    contrast_loss = -torch.log(pos_samples / (neg_samples + 1e-8)).mean()
    
    # 组合损失（可根据任务调整权重）
    return mse_loss + 0.4 * contrast_loss, mse_loss, contrast_loss

# 评估函数
def evaluate(model, loader, temperature):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            pred, emb1, emb2 = model(**inputs)

            loss, mse, contrast = hybrid_loss(pred, labels, emb1, emb2, temperature)
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # print(len(all_preds), len(all_labels))
    # print(all_preds[0:30], all_labels[0:30])
    spearman_corr = spearmanr(all_preds, all_labels).correlation
    return total_loss / len(loader), spearman_corr

def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    # 初始化模型和分词器
    model_name = "openai/clip-vit-large-patch14-336"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPPairModel(model_name).to(device)
    
    # 加载数据集（示例使用STS-B，实际替换为自定义数据）
    dataset = load_dataset("glue", "stsb")
    train_data = SentencePairDataset(tokenizer, dataset['train'])
    val_data = SentencePairDataset(tokenizer, dataset['validation'])
    test_data = SentencePairDataset(tokenizer, dataset['test'])

    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step / args.warmup, 1.0) if args.warmup > 0 else 1.0
    )

    best_corr = -1.0
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            print('inputs', inputs['input_ids1'],len(inputs['input_ids1']))
            optimizer.zero_grad()
            pred, emb1, emb2 = model(**inputs)
            # print(emb1[0][0:10])
            # print(emb1[1][0:10])
            loss, mse, contrast = hybrid_loss(
                pred, labels, emb1, emb2, model.temperature
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse.item():.4f}",
                "contrast": f"{contrast.item():.4f}"
            })

        # 验证集评估
        val_loss, val_corr = evaluate(model, val_loader, model.temperature)
        print(f"\nValidation Loss: {val_loss:.4f} | Spearman: {val_corr:.4f}")
        
        # 保存最佳模型
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pth")
            print("Saved new best model!")

    # 最终测试
    model.load_state_dict(torch.load("best_model_epoch5.pth"))
    test_loss, test_corr = evaluate(model, test_loader, model.temperature)
    print(f"\nTest Performance: Loss={test_loss:.4f} | Spearman={test_corr:.4f}")

if __name__ == "__main__":
    
    main()


