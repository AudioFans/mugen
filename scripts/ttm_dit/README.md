# Text to Music

Here are DiT based music generation training. The DAC latents are used as representations. 

Export the paths of audidata and mugen of python before training, e.g.

```bash
export PYTHONPATH=$PYTHONPATH:/xxx/audidata/:/yyy/mugen
```

## Single card train
```python
CUDA_VISIBLE_DEVICES=1 python scripts/ttm/train_dit_gtzan.py
```

## Multiple GPUs train

```python
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --multi_gpu --num_processes 4 scripts/ttm/train_dit_gtzan_accelerate.py
```

## Generate
```python
CUDA_VISIBLE_DEVICES=2 python scripts/ttm/generate_dit_gtzan.py
```