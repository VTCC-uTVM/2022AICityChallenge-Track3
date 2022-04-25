import pandas
# df = pd.read_csv('total_data.csv')


k_map = {"dashboard": "Dashboard", "rearview": "Rear", "right": "Right"}

for view in ["dashboard", "rearview", "right"]:
    for user_id in ["49381", "38058", "35133", "24491", "24026"]:
        f = open("total_data.csv", "r")
        f_train = open("data/train_{}_{}.csv".format(view, user_id), "w")
        f_val = open("data/val_{}_{}.csv".format(view, user_id), "w")
        f_test = open("data/test_{}_{}.csv".format(view, user_id), "w")
        for line in f.readlines():
            if k_map[view] in line:    
                if user_id in line:
                    print(line)
                    f_val.write(line)
                    f_test.write(line)
                else:
                    f_train.write(line)
        f_val.close()
        f_train.close()
        f_test.close()
        f.close()
