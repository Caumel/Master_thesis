import os


path = r"/media/data/lazarenkom98dm/coeffs-second-visit"
path1 = r"/media/data/lazarenkom98dm/coeffs-first-visit"

def vers_1(path):
    """
    Renaming the coefficients filenames with a specific pattern.
    """
    for file_name in os.listdir(path):
        splitted_filename = file_name.split('_')
        splitted_filename[4] = splitted_filename[4].replace('.txt.txt', '.txt')
        old_name = f"{path}\{file_name}"
        new_name = f'{path}\data_{splitted_filename[3]}_{splitted_filename[4]}'
        os.rename(old_name, new_name)

def vers_2(path1):
    """
    Renaming the coefficients filenames with a specific pattern.
    """
    for file_name in os.listdir(path1):
        splitted_filename = file_name.split('_')
        splitted_filename[2] = splitted_filename[2].replace('.txt.txt', '.txt')
        joined_filename = '_'.join(splitted_filename)
        old_name = f"{path1}\{file_name}"
        new_name = f'{path1}\{joined_filename}'
        os.rename(old_name, new_name)


vers_1(path)
vers_2(path1)

