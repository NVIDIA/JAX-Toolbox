import webdataset as wds
import sys

BASENAME='custom_eval_prompts'
with open(f'{BASENAME}.txt', 'r') as f:
    prompts = f.readlines()

sink = wds.TarWriter(f"{BASENAME}.tar")
for index, line in enumerate(prompts):
    if index%1000==0:
        print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
    sink.write({
        "__key__": "sample%06d" % index,
        "txt": line.strip(),
    })
sink.close()
