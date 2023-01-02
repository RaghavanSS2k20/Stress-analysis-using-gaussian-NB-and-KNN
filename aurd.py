import serial
import time
import pandas as pd
import csv 
from datetime import datetime

ser = serial.Serial("COM6",115200,timeout=1)
# time.sleep(2)
lst2 = []
tup = []
with open('data.csv','a',encoding='UTF8',newline='') as f:
    w = csv.writer(f)
# anotherZList
    for i in range(1000):
        line=ser.readline().decode("utf-8")
        print(line)
        if(line !=''):
            tup = []
            # tup.append(str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            tup.append(i for i in line)
            w.writerow(tup)
