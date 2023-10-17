with open('meta_info/meta_info_Vimeo90K_train_GT.txt', 'r') as fin:
    keys = [line for line in fin]
f = open('meta_info/meta_crop_info_Vimeo90K_train_GT.txt', 'a')
for i in range(2):
    count = 0
    for j in keys:
        count+=1
        #print(count)
        #print(str(i) + '/' + j + '\n')
        f.write(str(i) + '/' + j)
