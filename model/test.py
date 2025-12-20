# Fichier de test rapide (test_pipeline.py)
import torch
from model import SOTARealEstateModel
from data_loader import get_cols_config, prepare_preprocessors, RealEstateDataset, real_estate_collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def sanity_check():
    # 1. Config
    csv_folder = "./data/train" # Ton dossier CSV
    img_folder = "./data/images" # Ton dossier Images
    
    cont_cols, cat_cols, _ = get_cols_config()
    
    # 2. Calibration (Simulation)
    # Assure-toi d'avoir au moins un CSV dans le dossier pour tester
    scaler, medians, modes, cat_mappings, cat_dims = prepare_preprocessors(csv_folder, cont_cols, cat_cols)
    
    # 3. Tokenizer HF
    hf_tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')
    
    # 4. Dataset & Loader
    ds = RealEstateDataset(csv_folder, img_folder, hf_tokenizer, scaler, medians, modes, cat_mappings, cont_cols, cat_cols)
    dl = DataLoader(ds, batch_size=2, collate_fn=real_estate_collate_fn)
    
    # 5. Instanciation Modèle
    model = SOTARealEstateModel(
        num_continuous=len(cont_cols),
        cat_cardinalities=cat_dims,
        img_model_name='resnet18', # Plus léger pour le test
        text_model_name='almanach/camembert-base'
    )
    
    # 6. Test d'un batch
    batch = next(iter(dl))
    print("Clés du batch :", batch.keys())
    
    # PASSAGE DANS LE MODÈLE
    # On unpack le dictionnaire directement dans les arguments nommés du forward
    # C'est LA clé de la cohérence : les noms dans collate_fn doivent matcher forward()
    outputs = model(
        images=batch['images'],
        image_masks=batch['image_masks'],
        input_ids=batch['input_ids'],
        text_mask=batch['text_mask'],
        x_cont=batch['x_cont'],
        x_cat=batch['x_cat']
    )
    
    pred_vente, pred_loc, _ = outputs
    print(f"Sortie Vente shape: {pred_vente.shape}") # Doit être [2, 1]
    print("Sanity Check : SUCCÈS TOTAL")

if __name__ == "__main__":
    sanity_check()
