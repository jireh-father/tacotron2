import os
import glob
import pickle

encoder_dataset_dir = "/home/ilseo/dataset/encoder_preprocessing/"
target_dataset_prefixs = ["datatang", "zeroth-korean", "speech"]
output_path = "/home/ilseo/dataset/speaker_id_map_datatang_zeroth_speechko.pkl"

target_spk_dirs = []
for target_dataset_prefix in target_dataset_prefixs:
    target_spk_dirs += glob.glob(os.path.join(encoder_dataset_dir, target_dataset_prefix) + "_*")

file_spk_id_map = {}
spk_cnt = 0
for spk_id, target_spk_dir in enumerate(target_spk_dirs):
    file_names = glob.glob(os.path.join(target_spk_dir, "*.npy"))
    if len(file_names) > 0:
        spk_cnt += 1
    for file_name in file_names:
        file_name = os.path.splitext(os.path.basename(file_name))[0]
        if file_name in file_spk_id_map:
            print("dup!!!", file_name, target_spk_dir)
            raise Exception("dup!!")
        file_spk_id_map[file_name] = spk_id
with open(output_path, 'wb+') as f:
    pickle.dump(file_spk_id_map, f)
print(spk_cnt, "speakers completed.")
