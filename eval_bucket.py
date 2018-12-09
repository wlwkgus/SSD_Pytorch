import argparse
import operator
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str, help='file path')

args = parser.parse_args()

file_path = args.file_path

ap_dict = dict()
with open(file_path, 'r') as f:
    lines = f.readlines()

    for line in lines:
        if line.startswith('AP for'):
            ws = line.strip().split(' ')
            name = ' '.join(ws[2:-2])
            ap_score = float(ws[-1])
            ap_dict[name] = ap_score

sorted_ap_dict_items = sorted(ap_dict.items(), key=operator.itemgetter(1), reverse=True)
print("[ Top 30 ]")

for tup in sorted_ap_dict_items[:30]:
    print("{} : {}".format(tup[0], tup[1]))

top_array = np.asarray([tup[1] for tup in sorted_ap_dict_items])
print(">>> Top 30 mean AP : {}".format(top_array.mean()))
print(">>> Top 30 stddev AP : {}".format(top_array.std()))

sorted_ap_dict_items = sorted(ap_dict.items(), key=operator.itemgetter(1))
print("[ Worst 30 ]")

for tup in sorted_ap_dict_items[:30]:
    print("{} : {}".format(tup[0], tup[1]))

top_array = np.asarray([tup[1] for tup in sorted_ap_dict_items])
print(">>> Worst 30 mean AP : {}".format(top_array.mean()))
print(">>> Worst 30 stddev AP : {}".format(top_array.std()))
