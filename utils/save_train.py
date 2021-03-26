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

def save_config_pro(save_path):

    train_config_save_dir = save_path + '/training_configuration'
    os.makedirs(train_config_save_dir)
    current_path = os.getcwd()
    files = os.listdir(current_path)
    for file in files:
        if file.endswith('.py') or file.endswith('.sh'):
            file_path = os.path.join(current_path,file)
            shutil.copy2(file_path,train_config_save_dir)

    # Copy Folders
    shutil.copytree('./utils', os.path.join(train_config_save_dir, 'utils'))
    shutil.copytree('./analyse', os.path.join(train_config_save_dir, 'analyse'))
    shutil.copytree('./single_input_VAE', os.path.join(train_config_save_dir, 'single_input_VAE'))


