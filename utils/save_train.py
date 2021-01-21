import os, shutil
from datetime import datetime

def create_log_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    now = datetime.now()
    if not save_dir.endswith('/'):
        save_dir += '/'
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    train_data_dir = save_dir + dt_string
    os.makedirs(train_data_dir)
    return train_data_dir

def save_train_config(*file_list, save_path):
    if not file_list:
        pass
    else:
        train_config_dir = save_path + '/training_configuration'
        os.makedirs(train_config_dir)
        for file in file_list:
            shutil.copy2(file, train_config_dir)

