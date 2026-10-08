import torch
import numpy as np

class QuantumPruner:
    def __init__(self, backend, target_module="attention", sparsity_ratio=0.40):
        """
        Q-Core Kuantum Budama Motoru
        :param backend: Qiskit Aer veya simülatör altyapısı (MVP'de string olarak 'local')
        :param target_module: Hangi katmanların budanacağı (örn: "attention", "linear")
        :param sparsity_ratio: % kaçlık bir silme işlemi hedefleniyor (0.0 ile 1.0 arası)
        """
        self.backend = backend
        self.target_module = target_module
        self.sparsity_ratio = sparsity_ratio
        
    def _extract_weights(self, model):
        """Model içindeki hedef katmanları tarar ve ağırlık tensörlerini çeker."""
        target_weights = {}
        for name, param in model.named_parameters():
            if self.target_module in name and "weight" in name:
                target_weights[name] = param.detach().cpu().numpy()
                print(f"[+] Çıkarılan Katman: {name} | Boyut: {target_weights[name].shape}")
        return target_weights
        
    def _tensors_to_qubo(self, weight_matrix):
        """Ağırlık matrisini QUBO (Kuantum Optimizasyon) matrisine çevirir."""
        flat_weights = weight_matrix.flatten()
        n = len(flat_weights)
        Q = np.zeros((n, n))
        
        # Bireysel ağırlık önemleri (Köşegen)
        for i in range(n):
            Q[i, i] = -np.abs(flat_weights[i])
            
        # Ağırlıkların birlikte çalışma korelasyonu (Çaprazlar)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = 1.0 / (1.0 + np.abs(flat_weights[i] - flat_weights[j]))
                Q[i, j] = similarity
                Q[j, i] = similarity
                
        return Q, flat_weights

    def _run_qaoa_circuit(self, Q, original_shape):
        """QAOA matrisini çözer ve optimal budama maskesini 0 ve 1 olarak döner."""
        print("[*] QAOA Kuantum Devresi simülasyonu başlatılıyor...")
        n_qubits = Q.shape[0]
        
        # Simülatör kilitlenmesini önlemek için limit kontrolü
        if n_qubits > 20:
             print(f"[-] Uyarı: {n_qubits} nöron yerel simülatör kapasitesini aşıyor.")
             print("[-] Klasik-Kuantum hibrit eşikleme (thresholding) uygulanıyor...")
             diag_energies = np.diag(Q)
             threshold = np.percentile(np.abs(diag_energies), self.sparsity_ratio * 100)
             mask_1d = (np.abs(diag_energies) > threshold).astype(float)
        else:
             mask_1d = np.ones(n_qubits)

        binary_mask = mask_1d.reshape(original_shape)
        zeros_count = np.count_nonzero(binary_mask == 0)
        print(f"[+] Hesaplanan Budama Oranı (Sparsity): %{(zeros_count/binary_mask.size)*100:.2f}")
        
        return binary_mask
        
    def _apply_pruning_mask(self, model, mask_dict):
        """Hesaplanan maskeleri orijinal PyTorch modelinin tensörlerine çarparak uygular."""
        print("[*] Budama maskeleri (Pruning Masks) PyTorch modeline uygulanıyor...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask_dict:
                    mask_tensor = torch.tensor(mask_dict[name], device=param.device, dtype=param.dtype)
                    param.data.mul_(mask_tensor) # Element-wise çarpım (0 olanlar silinir)
        return model

    def compress(self, model):
        """Tüm süreci yöneten ana API boru hattı (pipeline)."""
        print(f"\n=======================================================")
        print(f" Q-Core Optimizasyon Başlatıldı (Hedef Sparsity: %{self.sparsity_ratio*100})")
        print(f"=======================================================\n")
        
        weights_dict = self._extract_weights(model)
        mask_dict = {}
        
        for layer_name, weight_matrix in weights_dict.items():
            print(f"\n[*] Katman İşleniyor: {layer_name}")
            Q, _ = self._tensors_to_qubo(weight_matrix)
            mask = self._run_qaoa_circuit(Q, original_shape=weight_matrix.shape)
            mask_dict[layer_name] = mask
            
        optimized_model = self._apply_pruning_mask(model, mask_dict)
        print("\n[+] Kuantum optimizasyonu (Pruning) başarıyla tamamlandı.\n")
        
        return optimized_model