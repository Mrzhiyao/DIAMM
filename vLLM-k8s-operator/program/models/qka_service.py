import asyncio
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
import torch
import torch.nn as nn
import json
from transformers import CLIPTokenizer, CLIPTextModel

# ===================== 全局配置 =====================
HOST = "192.168.2.75"
PORT = 7889
MAX_WORKERS = 8  # 根据GPU能力调整
MODEL_PATH = "/yaozhi/vLLM-k8s-operator/program/dis_model.pth"

# ===================== 模型定义 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

class RelevanceModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # 解冻模型最后两层
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for i in [-2, -1]:
            for param in self.text_encoder.text_model.encoder.layers[i].parameters():
                param.requires_grad = True

        # 分类头部
        hidden_size = self.text_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(1.5, requires_grad=True))

    def _masked_pooling(self, hidden_states, attention_mask):
        weights = attention_mask.unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        out1 = self.text_encoder(input_ids1, attention_mask1).last_hidden_state
        out2 = self.text_encoder(input_ids2, attention_mask2).last_hidden_state
        
        emb1 = self._masked_pooling(out1, attention_mask1)
        emb2 = self._masked_pooling(out2, attention_mask2)
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat([emb1, emb2, diff], dim=1)
        output = self.classifier(combined).squeeze()
        return output * self.scale

# ===================== 模型服务 =====================
model = RelevanceModel("openai/clip-vit-base-patch32").to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

print(f"模型加载完成，设备: {device}, 线程数: {MAX_WORKERS}")

def _sync_predict(text1: str, text2: str) -> float:
    """同步推理函数"""
    with torch.no_grad():
        inputs = tokenizer(
            [text1, text2],
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids1 = inputs['input_ids'][0].unsqueeze(0).to(device)
        attention_mask1 = inputs['attention_mask'][0].unsqueeze(0).to(device)
        input_ids2 = inputs['input_ids'][1].unsqueeze(0).to(device)
        attention_mask2 = inputs['attention_mask'][1].unsqueeze(0).to(device)
        
        raw_score = model(
            input_ids1=input_ids1,
            attention_mask1=attention_mask1,
            input_ids2=input_ids2,
            attention_mask2=attention_mask2
        ).item()
        
        return raw_score * (1.5 / model.scale.item())

async def predict_handler(request):
    """处理预测请求"""
    try:
        data = await request.json()
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        
        if not text1 or not text2:
            return web.json_response(
                {'error': '缺少 text1 或 text2 参数'},
                status=400
            )
        
        loop = asyncio.get_event_loop()
        # 提交到线程池执行
        score = await loop.run_in_executor(
            executor,
            _sync_predict,
            text1, text2
        )
        
        return web.json_response({
            'text1': text1,
            'text2': text2,
            'score': round(score, 4),
            'threshold': 1.0  # 分类阈值参考值
        })
    
    except Exception as e:
        return web.json_response(
            {'error': f'内部错误: {str(e)}'},
            status=500
        )

async def health_handler(request):
    """健康检查端点"""
    return web.json_response({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True
    })

async def init_app():
    """初始化应用"""
    app = web.Application()
    app.router.add_post('/predict', predict_handler)
    app.router.add_get('/health', health_handler)
    return app

if __name__ == '__main__':
    print(f"启动服务: http://{HOST}:{PORT}")
    web.run_app(init_app(), host=HOST, port=PORT)
