"""
Transform the original SST-2 dataset to GLUE style
"""
for f in ['train.tsv', 'dev.tsv']:
    tar = open('tar-' + f, 'w')
    tar.write('sentence\tlabel\n')
    for line in open(f).readlines():
        label = line[0]
        tar.write(line[2:].rstrip() + '\t' + label + '\n')
