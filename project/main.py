import os
from sqlite3 import DatabaseError
import pandas
if __name__=="__main__":
    
    path = "../images"

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
    targets=[]
    for i in range(0,len(galaxys)):
        targets.append(i)


    df=pandas.DataFrame(columns=["nameImg","id"])
    df=pandas.DataFrame(list(zip(galaxys,targets)), columns=["nameImg","id"])

    print(df)


