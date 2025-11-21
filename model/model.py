import argparse
import os
import time
import torch
import platform
import psutil
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from pathlib import Path



def format_bytes(num_bytes: int) -> str:
    """Formatte les octets en MB/GB lisibles."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def system_info(verbose=True):
    """Affiche les infos système / PyTorch / CUDA."""
    if not verbose:
        return
    print("===== Informations système =====")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capacité mémoire totale: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
    print("===================================")


def gpu_memory(label: str):
    """Affiche la mémoire GPU utilisée."""
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0)
        mem_reserved = torch.cuda.memory_reserved(0)
        print(f"Mémoire GPU ({label}) : {format_bytes(mem_used)} utilisée / {format_bytes(mem_reserved)} réservée")


def load_model(model_id: str, device, verbose: bool = False):
    start = time.time()
    print(f"\n Chargement du modèle {model_id} avec dtype=float32, device={device} ...")

    processor = Blip2Processor.from_pretrained(model_id)

    try:
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )
        if verbose:
            print("→ Modèle chargé avec device_map='auto'.")
    except Exception as e:
        if verbose:
            print(f"device_map='auto' impossible ({e}), fallback simple .to(device)")
        model = Blip2ForConditionalGeneration.from_pretrained(model_id)
        model.to(device)

    # Forcer float32
    model = model.to(torch.float32).to(device)
    model.eval()

    try:
        if verbose: print(" Compilation torch.compile() ...")
        model = torch.compile(model)
        if verbose: print("→ Compilation réussie.")
    except Exception as e:
        if verbose: print(f"torch.compile échoué : {e}")

    load_time = time.time() - start
    print(f"✅ Modèle chargé en {load_time:.2f} s")
    return processor, model, load_time


def generate_caption(processor, model, image_path, prompt, device, max_tokens, verbose):
    start = time.time()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Forcer float32 pour les entrées
    inputs = {k: v.to(device=device, dtype=torch.float32) if v.dtype.is_floating_point else v.to(device) for k, v in inputs.items()}

    gpu_memory("avant génération")

    with torch.inference_mode():
        generated_ids =generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )


    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    gpu_memory("après génération")

    duration = time.time() - start
    if verbose:
        print(f"{image_path.name} traité en {duration:.2f}s → {decoded}")
    return decoded, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="../output")
    parser.add_argument("--prompt", type=str, default="Question: Tell me in detail what does the image depict? Answer:")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Infos système
    system_info(args.verbose)

    # Device GPU
    device = torch.device("cuda")

    # Mémoire CPU avant
    cpu_mem_before = psutil.Process(os.getpid()).memory_info().rss
    if args.verbose:
        print(f" Mémoire RAM avant chargement: {format_bytes(cpu_mem_before)}")

    # Chargement modèle
    model_id = "Salesforce/blip2-opt-2.7b"
    processor, model, load_time = load_model(model_id, device, args.verbose)
    gpu_memory("après chargement")

    # Liste d'images
    image_dir = Path(args.image_dir)
    images = [image_dir] if image_dir.is_file() else [p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    if not images:
        print("Aucune image trouvée.")
        return

    print(f"\n{len(images)} image(s) trouvée(s). Génération en cours...\n")

    total_time = 0
    results = {}

    for img in images:
        caption, duration = generate_caption(processor, model, img, args.prompt, device, args.max_new_tokens, args.verbose)
        results[img.name] = caption
        total_time += duration

    cpu_mem_after = psutil.Process(os.getpid()).memory_info().rss
    print("\n===== RÉSUMÉ FINAL =====")
    print(f"Temps de chargement modèle : {load_time:.2f} s")
    print(f"Temps total d'inférence : {total_time:.2f} s ({total_time/len(images):.2f} s/image)")
    print(f"Mémoire RAM utilisée : {format_bytes(cpu_mem_after - cpu_mem_before)}")
    gpu_memory("final")
    print("============================\n")

    for name, caption in results.items():
        print(f"{name} → {caption}")


if __name__ == "__main__":
    main()
