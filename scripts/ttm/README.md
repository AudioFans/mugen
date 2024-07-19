# Text to Music

Here are language model based training. The first level DAC codec is converted to tokens. For a 10-second audio there are around 870 tokens.

Export the paths of audidata and mugen of python before training, e.g.

```bash
export PYTHONPATH=$PYTHONPATH:/xxx/audidata/:/yyy/mugen
```

## Single card train
```python
CUDA_VISIBLE_DEVICES=1 python scripts/ttm/train_lm_gtzan.py --model_name=llama
```

## Multiple GPUs train

```python
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --multi_gpu --num_processes 4 scripts/ttm/train_lm_gtzan_accelerate.py
```

## Generate
```python
CUDA_VISIBLE_DEVICES=2 python scripts/ttm/generate_lm_gtzan.py
```

## Results

<img src="assets/ttm/train_lm_gtzan.png" width="800">