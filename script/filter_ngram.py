import sys

input_file_path = sys.argv[1]
min_ngram_len = int(sys.argv[2])
min_ngram_count = int(sys.argv[3])

with open(input_file_path) as i:
    with open(f'{input_file_path}-filter', 'w') as o1, open(
            f'{input_file_path}-filter-text', 'w') as o2:
        for line in i:
            line = line.replace('<s>', '</s>')
            ngram, count = line.strip().split('\t')
            if len(ngram.split()) < min_ngram_len:
                continue
            if int(count) < min_ngram_count:
                continue
            o1.write(line)
            o2.write(f'{ngram}\n')
