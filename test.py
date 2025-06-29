import zipfile

folder_path = 'D:/Projects/SOAI/auction_2025/auction.egg'
folder_path2 = 'D:/Projects/SOAI/auction_2025/'
               
with zipfile.ZipFile(folder_path, "r") as egg_file:
    egg_file.extractall(folder_path2)
