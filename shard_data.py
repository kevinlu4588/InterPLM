# split_fasta_round_robin.py
from pathlib import Path
from Bio import SeqIO
import gzip

src_gz   = Path("/home/ec2-user/SageMaker/InterPLM/data/uniprot/uniprot_sprot.fasta.gz")  # copy here first
out_dir  = Path("/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot/")
WORLD    = 8  # match torchrun --nproc_per_node

out_dir.mkdir(parents=True, exist_ok=True)

outs = [open(out_dir / f"shard_{i:02d}.fasta", "w") for i in range(WORLD)]
try:
    with gzip.open(src_gz, "rt") as fh:
        for i, rec in enumerate(SeqIO.parse(fh, "fasta")):
            k = i % WORLD
            outs[k].write(f">{rec.id}\n{str(rec.seq)}\n")
finally:
    for f in outs:
        f.close()
print("Done sharding.")
