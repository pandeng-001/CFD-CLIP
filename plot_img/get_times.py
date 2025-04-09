# Get model training time
import os


file_path = "./log/(CLIP)WRN-16-8/txt/2-(CLIP)WRN-16-8-classifier__time_mins.txt"

# print(os.path.exists(file_path))

with open(file_path, "r") as f:
    time_data = f.readlines()
    # print(len(time_data))
    time_init = 0
    time_list = time_data[0].split(",")
    for i in time_list:
        if i != " \n":
            time_init += float(i)

print("Total training duration: {:.4f}h".format(time_init/60))
    