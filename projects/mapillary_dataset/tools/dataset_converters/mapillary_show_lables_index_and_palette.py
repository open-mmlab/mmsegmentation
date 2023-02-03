import json

version = 'v1.2'
# version = 'v2.0'

with open('../../../../data/mapillary/config_{}.json'.format(
        version)) as config_file:
    config = json.load(config_file)
# in this example we are only interested in the labels
labels = config['labels']

print(f'There are {len(labels)} labels classes in {version}')

for label_id, label in enumerate(labels):
    print(f'{label_id}--{label["readable"]}--{label["color"]}, ')
