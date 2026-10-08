import torch
import torch.nn as nn
from q_core import QuantumPruner

# Sistemin çalıştığını görmek için oluşturulmuş minik bir Yapay Zeka modeli
class DummyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # 'attention' kelimesini özellikle koyduk ki QuantumPruner bunu bulup hedef alsın
        self.attention_weight_1 = nn.Linear(10, 10) 
        self.attention_weight_2 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, 2)

    def forward(self, x):
        x = self.attention_weight_1(x)
        x = self.attention_weight_2(x)
        return self.output_layer(x)

if __name__ == "__main__":
    print("[1] Test Modeli Belleğe Yükleniyor...")
    model = DummyLLM()
    
    # Kuantum Budama motorunu başlatıyoruz (Sadece attention katmanlarını %40 budayacak)
    pruner = QuantumPruner(backend="local_simulator", target_module="attention", sparsity_ratio=0.40)
    
    # Modeli sıkıştır
    optimized_model = pruner.compress(model)
    
    print("[2] Optimizasyon sonrası ağırlıkların örneği (0 olanlar budananlardır):")
    # Budanan katmanın ilk satırındaki ağırlıkları ekrana basıp 0'ları gözümüzle görelim
    print(optimized_model.attention_weight_1.weight.data[0])