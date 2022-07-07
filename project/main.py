import os

if __name__=="__main__":
    
    path = "../images"
    # change the working directory to the path where the images are located
    os.chdir(path)

    # this list holds all the image filename
    galaxys = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
    # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
            # adds only the image files to the flowers list
                galaxys.append(file.name)

    print(galaxys)