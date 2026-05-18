import os
import joblib
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_model(cfg: DictConfig):
    # ... tu código de entrenamiento ...
    
    #  Obtiene la ruta de donde lanzaste el script (la raíz del proyecto)
    root_dir = hydra.utils.get_original_cwd()
    
    # Construimos la ruta final: /tu_proyecto/models/modelo_airbnb_ganador.pkl
    save_path = os.path.join(root_dir, cfg.paths.models_dir, cfg.paths.model_name)
    
    # Nos aseguramos de que la carpeta existat antes de guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Guardamos el artefacto
    joblib.dump(modelo_final, save_path)
    print(f" Modelo de producción actualizado en: {save_path}")

if __name__ == "__main__":
    train_model()
