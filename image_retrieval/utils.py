import os

def load_data(path):
    img_paths = []
    img_labels = []
    folder_lst = os.listdir(path)
    for i, folder in enumerate(folder_lst):
        folder_path = os.path.join(path, folder)
        file_lst = os.listdir(folder_path)
        for file in file_lst:
            img_paths.append(os.path.join(folder_path, file))
            img_labels.append(i)
    return img_paths, img_labels