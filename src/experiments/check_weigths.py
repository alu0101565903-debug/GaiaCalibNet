import joblib
import torch

def check_model_health():
    model_path = "results/models/pinn/pinn_spectrum.joblib"
    print(f"Abriendo el cerebro de la IA: {model_path}\n")
    
    checkpoint = joblib.load(model_path)
    state_dict = checkpoint.get('model_state_dict')
    
    if not state_dict:
        print("[!] No se encontraron pesos.")
        return

    is_corrupted = False
    for name, param in state_dict.items():
        max_val = torch.max(param).item()
        min_val = torch.min(param).item()
        has_nan = torch.isnan(param).any().item()
        
        print(f"Capa: {name}")
        print(f"  -> Valor Máximo: {max_val:.4f}")
        print(f"  -> Valor Mínimo: {min_val:.4f}")
        
        if has_nan or abs(max_val) > 100 or abs(min_val) > 100:
            print("  [!] ADVERTENCIA: Valores anómalos detectados.")
            is_corrupted = True
        print("-" * 40)
        
    if is_corrupted:
        print("\n[DIAGNÓSTICO] Tu modelo ha sufrido una explosión de gradientes.")
    else:
        print("\n[DIAGNÓSTICO] Los pesos parecen estar en un rango normal.")

if __name__ == "__main__":
    check_model_health()