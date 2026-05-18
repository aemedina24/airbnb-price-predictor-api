import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Extraer las listas del YAML de forma segura
    # Convertimos a listas nativas de Python usando OmegaConf.to_container
    text_features = OmegaConf.to_container(cfg.features.text)
    numeric_features = OmegaConf.to_container(cfg.features.numeric)
    binary_features = OmegaConf.to_container(cfg.features.binary)
    target = cfg.target

    # 2. Consolidar todas las columnas que necesitamos leer
    cols_to_read = text_features + numeric_features + binary_features + [target]

    # 3. Leer el dataset
    df = pd.read_csv(
        cfg.data.path,
        usecols=cols_to_read
    )

    print("--- Primeras filas del Dataset ---")
    print(df.head())
    print("\nShape total del DataFrame:", df.shape)


if __name__ == "__main__":
    main()
