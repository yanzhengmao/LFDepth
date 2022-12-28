path  = '/home/yzm/eigen_train_files_with_gt.txt'

with open(path,'r') as f:
    filename = f.readlines()
f.close()
n = len(filename)
data_path = '/home/yzm/project/TransDepth-main/train_test_inputs/new_eigen_train_files_with_gt.txt'
with open(data_path,'w') as f:
    for i in range(n):
        b = filename[i].split(' ')
        b.insert(1, b[0].replace('image_02', 'image_03'))
        new_path = ' '.join(b)
        f.write(new_path)
f.close()

eval_path = '/home/yzm/project/TransDepth-main/train_test_inputs/eigen_test_files_with_gt.txt'
with open(eval_path,'r') as f:
    filename = f.readlines()
f.close()
n = len(filename)
data_eval_path = '/home/yzm/project/TransDepth-main/train_test_inputs/new_eigen_test_files_with_gt.txt'
with open(data_eval_path,'w') as f:
    for i in range(n):
        b = filename[i].split(' ')
        b.insert(1, b[0].replace('image_02', 'image_03'))
        new_path = ' '.join(b)
        f.write(new_path)
f.close()

