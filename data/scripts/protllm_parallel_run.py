import os, sys
import torch
import torch.multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from argparse import Namespace
from protllm.model.encoder import EasyProtSt
import esm
from queue import Empty


def load_model(device):
    model_args = Namespace(
        esm_model_name="esm2_t33_650M_UR50D",
        esm_model_file_path="",
        esm_tok_arch_name=esm.Alphabet.from_architecture("ESM-1b")
    )
    model = EasyProtSt(model_args)
    model.eval().to(device)
    return model, model_args.esm_tok_arch_name


def embed_fasta(fasta_path, output_csv, model, tokenizer, device):
    protein_ids, embeddings = [], []
    records = list(SeqIO.parse(fasta_path, "fasta"))
    with torch.no_grad():
        for record in tqdm(records, desc=os.path.basename(fasta_path), leave=False):
            tokens = tokenizer.encode(str(record.seq))
            mask = [1] * len(tokens)
            emb = model(
                torch.tensor([tokens]).to(device),
                torch.tensor([mask]).float().to(device)
            ).cpu().squeeze().numpy()
            protein_ids.append(record.id)
            embeddings.append(emb)
    df = pd.DataFrame(embeddings)
    df.insert(0, "protein_id", protein_ids)
    df.to_csv(output_csv, index=False)
    print(f"[Worker {os.getpid()}] Saved: {output_csv}")


def worker(rank, gpu_id, task_queue, seq_dir, out_dir):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {rank}] Starting on {device}")
    model, tokenizer = load_model(device)

    while True:
        try:
            filename = task_queue.get(timeout=10)
        except Empty:
            break

        stem = os.path.splitext(filename)[0]
        fasta_path = os.path.join(seq_dir, filename)
        output_csv = os.path.join(out_dir, f"{stem}_protein_embeddings.csv")

        if not os.path.exists(output_csv):
            try:
                embed_fasta(fasta_path, output_csv, model, tokenizer, device)
            except Exception as e:
                print(f"[Worker {rank}] Failed {filename}: {e}")

    print(f"[Worker {rank}] Done.")


def main():
    seq_dir = sys.argv[1]
    out_dir = os.path.join(seq_dir, "Embedded_Sequences")
    os.makedirs(out_dir, exist_ok=True)

    tasks = sorted(
        f for f in os.listdir(seq_dir)
        if f.endswith((".fa", ".fasta"))
        and not os.path.exists(
            os.path.join(out_dir, f"{os.path.splitext(f)[0]}_protein_embeddings.csv")
        )
    )

    if not tasks:
        print("No files to process.")
        return

    num_gpus = torch.cuda.device_count()
    num_workers = max(num_gpus, 1)
    print(f"Files to embed: {len(tasks)} | Workers: {num_workers} | GPUs: {num_gpus}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    for f in tasks:
        task_queue.put(f)

    processes = [
        ctx.Process(
            target=worker,
            args=(rank, rank % max(num_gpus, 1), task_queue, seq_dir, out_dir)
        )
        for rank in range(num_workers)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All done.")


if __name__ == "__main__":
    main()
