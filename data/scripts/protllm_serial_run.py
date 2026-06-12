import os, sys
import torch
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from argparse import Namespace
from protllm.model.encoder import EasyProtSt
import esm

# ==== Configuration ====
seq_dir = sys.argv[1]
out_dir = f"./{seq_dir}/Embedded_Sequences/"
os.makedirs(out_dir, exist_ok=True)

# ==== Initialize ProtLLM ====
model_args = Namespace(
    esm_model_name="esm2_t33_650M_UR50D",
    esm_model_file_path="",
    esm_tok_arch_name=esm.Alphabet.from_architecture("ESM-1b")
)
model = EasyProtSt(model_args)
model.eval().cuda()
tokenizer = model_args.esm_tok_arch_name

# ==== Function to embed all proteins from a .fa file ====
def embed_fasta(fasta_path, output_csv):
    protein_ids = []
    embeddings = []

    records = list(SeqIO.parse(fasta_path, "fasta"))
    with torch.no_grad():
        for record in tqdm(records, desc=f"Embedding {os.path.basename(fasta_path)}"):
            nid = record.id
            seq = str(record.seq)
            tokens = tokenizer.encode(seq)
            residue_mask = [1] * len(tokens)
            tokens_tensor = torch.tensor([tokens]).cuda()
            mask_tensor = torch.tensor([residue_mask]).float().cuda()
            emb = model(tokens_tensor, mask_tensor).cpu().squeeze().numpy()
            protein_ids.append(nid)
            embeddings.append(emb)

    # Save to CSV
    df = pd.DataFrame(embeddings)
    df.insert(0, "protein_id", protein_ids)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")

# ==== Process all .fa files in the Sequences folder ====
#import pandas as pd
#df = pd.read_csv('Microbes_with_all_data_with_files.tsv', sep='\t')

for filename in os.listdir('COMAPREM'):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        #bacteria_id = file.split(".")[0]
        _file = filename.replace('.fa', '')
        fasta_path = os.path.join(seq_dir, filename)
        output_csv = os.path.join(out_dir, f"{_file}_protein_embeddings.csv")

        if not os.path.exists(output_csv):  # Skip if already done
            try:
                embed_fasta(fasta_path, output_csv)
            except Exception as e:
                print(f"❌ Failed {filename}: {e}")