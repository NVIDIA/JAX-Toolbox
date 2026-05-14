## AlphaFold 3 input materials:

- `generate_af_parameters.py`: Generate AF's model synthetic parameters. This file will create a `random_weights.bin.zst` file. You can run it as:
```bash
python generate_af_parameters.py --repo /opt/alphafold --out-model-dir /opt/alphafold/model
```

- `generate_inputs.py`: This file generate fake amminoacidic sequences that can be used as input for AF. You run it as:
```bash
python generate_inputs.py --out-dir /opt/alphafold_input_data/inputs_sequences --lengths 512 1024 2048 3072 4096 5120
```

- `af3_inference_benchmark.py`: This is the benchmark code, for measuring GPU seconds, compilation time, execution time. You can run it as:
```bash
python af3_inference_benchmark.py  --input-json-path /opt/alphafold_input_data/inputs_sequences/bench_L02048.json -model-idr /opt/alphafold/model --output-json output.json
```
This will run the Evoformer model on a 2048 tokens input sequence.
