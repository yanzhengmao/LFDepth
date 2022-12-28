import os

path = '/home/yzm/flow_sence_datatset/additional/'
data_path = []
data = []
for file in os.listdir(path):
    for data_file in os.listdir(os.path.join(path, file)):
        if data_file[-3:] == 'png' or data_file[-3:] == 'pfm':
            data_path.append(os.path.join(file, data_file))
    data_path.sort()
    data.append(data_path)
    data_path = []
print(data[0][0])

# data_file = '/home/yzm/project/TransDepth-main/train_test_inputs/flow_sence_files'
# with open(data_file,'w') as f:
#     for i in range(len(data)):
#         for j in range(len(data[0])):
#             if data[0][j]