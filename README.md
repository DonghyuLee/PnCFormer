# PnCFormer

**Official code for the paper “Transformer-based prediction of dispersion relation and transmittance in phononic crystals.”**

This repository provides:
- Deterministic data generation via a 1D transfer-matrix method (TMM)
- Lightweight dataset verification/visualization (HDF5)
- **PnCFormer**: a sequence-to-frequency Transformer for predicting dispersion relation and transmittance

## Install
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate <env>
```

## Generate data
```bash
python -m data_generation.run
```

## Verify datasets (quick sanity plots)
```bash
python -m data_generation.visualize --data_dir data --pairs "CA" --sample_size 20000 --freq_size 1000 --indices 0:3
```

## Train PnCFormer
```bash
python -m surrogate.train   --data_dir data --pairs "CA"   --sample_size 20000 --freq_size 500   --target dispersion_relation_supercell   --batch_size 32 --max_seq_len 14   --epochs 50 --lr 1e-4 --min_lr 1e-6   --save_name PnCFormer_DR_v1
```

## Visualize predictions (PnCFormer)
```bash
# Dispersion (supercell)
python -m surrogate.visualize   --ckpt surrogate/ckpt/PnCFormer_DR_v1.pth   --data_dir data --pairs "CA"   --sample_size 20000 --freq_size 500   --target dispersion_relation_supercell   --indices 0:3

# Transmittance
python -m surrogate.visualize   --ckpt surrogate/ckpt/PnCFormer_TR_v1.pth   --data_dir data --pairs "CA"   --sample_size 20000 --freq_size 500   --target transmittance   --indices 10,20,30 --no_clamp_pi
```

## Minimal layout
```
analysis.py                 # TMM core
data_generation/            # dataset generation + visualize
surrogate/                  # PnCFormer, loaders, training(include evluation), visualize
data/                       # generated HDF5 (if you run data_generation)
```

## Citation
If you use this code, please cite the paper above.

