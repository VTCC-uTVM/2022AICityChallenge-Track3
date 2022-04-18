import pandas
# df = pd.read_csv('total_data.csv')
f = open("total_data.csv", "r")
f_train = open("train_dashboard_24026.csv", "w")
f_val = open("val_dashboard_24026.csv", "w")
f_test = open("test_dashboard_24026.csv", "w")
for line in f.readlines():
    if 'Dashboard' in line:    
        if '24026' in line:
            print(line)
            f_val.write(line)
            f_test.write(line)
        else:
            f_train.write(line)
f_val.close()
f.close()
f_train.close()
