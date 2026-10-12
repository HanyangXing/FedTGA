# FedTGA: Cross-Tier Heterogeneous Federated Learning via Statistical Generative Alignment

FedTGA is a heterogeneous federated learning framework for end–edge–cloud collaboration. 

## Project Structure

```text
├── main_fedtga.py           # Entry point
├── fedtga.py                # Core FedTGA algorithm
├── autoencoder_pretrained.py # Pre-trained autoencoder
├── data_loader.py           # Data loading and partitioning
├── datasets.py              # CIFAR dataset wrappers
├── model_zoo.py             # Heterogeneous models and projection heads
├── utils.py                 # Loss functions and utilities
```

## Environment Dependencies

- Python 3.9+ (recommended)
- PyTorch and torchvision with CUDA support
- numpy, scipy, scikit-learn, matplotlib, Pillow

## Quick Start

```bash
python main_fedtga.py --dataset cifar10 --partition_alpha 1.0 --comm_round 500
```

### Common Parameters

| Parameter | Description |
| --- | --- |
| `--dataset` | Dataset |
| `--client_number` | Number of clients |
| `--edge_number` | Number of edge servers |
| `--partition_alpha` | Dirichlet data heterogeneity |
| `--comm_round` | Communication rounds |
| `--batch_size` | Local training batch size |
| `--samples_per_class` | Synthetic samples per class |
| `--con_temperature` | Contrastive temperature |
