import json
import os

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f
base = './outputs_full'
data = []
file_names = sorted(findAllFile(base))
for file_name in file_names:
    data += [json.loads(line.strip()) for line in open(os.path.join(base, file_name))]
    # data.update(json.load(open(os.path.join(base, file_name))))
print(len(data))
json.dump(data, open(os.path.join(base, 'predictions.json'), 'w'), indent = 2)