# This is the master code that will run in the OPi during actual prototype testing.
import pandas as pd
import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import metrics
import time
from time import sleep
import pandas as pd
import numpy as np
import re
import os
import sys
import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
import digitalio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import ili9341
from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

# variable to check for presence detection after person sits down 
isRisingPresence = False

isFirstRun = True

# filesave location
folderPathOPi = '/home/orangepi/opi-setup/buttCushion/'
imagePathOPi = '/home/orangepi/opi-setup/pics/'

# file name uniquifyer
fileLabelCounter = 1 
# Load the saved decision tree model
forestFile = 'decision_forest_model.joblib'
forestFile1 = 'decision_forest.joblib'
# clf = pickle.load(open(folderPathOPi + forestFile, 'rb'))
clf = joblib.load(folderPathOPi + forestFile1) # change if needed

# Configuration for CS and DC pins:
cs_pin = digitalio.DigitalInOut(board.PC11)
dc_pin = digitalio.DigitalInOut(board.PC6)
reset_pin = digitalio.DigitalInOut(board.PC9)
# Configuration for touch
touch_pin = digitalio.DigitalInOut(board.PC14)

# Config for display baudrate (default max is 24mhz):
BAUDRATE = 24000000
# Setup SPI bus using hardware SPI:
spi = board.SPI()
disp = ili9341.ILI9341(
    spi,
    rotation=90,  
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
)

# Create blank image for drawing.
# Make sure to create image with mode 'RGB' for full color.
if disp.rotation % 180 == 90:
    height = disp.width  # we swap height/width to rotate it to landscape!
    width = disp.height
else:
    width = disp.width 
    height = disp.height
image = Image.new("RGB", (width, height))
# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)
# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
disp.image(image)

# scales, crops and centers the image for display
def image_prep(filename): 
    image = Image.open(filename)
    # Scale the image to the smaller screen dimension
    image_ratio = image.width / image.height
    screen_ratio = width / height
    if screen_ratio < image_ratio:
        scaled_width = image.width * height // image.height
        scaled_height = height
    else:
        scaled_width = width
        scaled_height = image.height * width // image.width
    image = image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)

    # Crop and center the image
    x = scaled_width // 2 - width // 2
    y = scaled_height // 2 - height // 2
    image = image.crop((x, y, x + width, y + height))
    return image

# prepare images for display
image_cali_1 = image_prep(imagePathOPi + "Adjust Pose to be _good__1.png")
image_cali_2 = image_prep(imagePathOPi + "Adjust Pose to be _good__2.png")
image_cali_3 = image_prep(imagePathOPi + "Adjust Pose to be _good__3.png")
image_cali_4 = image_prep(imagePathOPi + "Adjust Pose to be _good__4.png")
image_cali_5 = image_prep(imagePathOPi + "Adjust pose to be _good__5.png")
image_cali_fail = image_prep(imagePathOPi + "Calibration Fail.png")
image_cali_success = image_prep(imagePathOPi + "Calibration Success.png")
image_cali_start = image_prep(imagePathOPi + "Calibration in Progress.png")
image_cushion_placement = image_prep(imagePathOPi + "How to place cushion.png")
image_reading = image_prep(imagePathOPi + "reading.png")
image_no_presence = image_prep(imagePathOPi + "Idle.png")
image_yes_presence = image_prep(imagePathOPi + "Presence Detected.png")
image_bad_pos_pred = image_prep(imagePathOPi + "Bad Posture.png")
image_good_pos_pred = image_prep(imagePathOPi + "Good posture.png")
image_connecting = image_prep(imagePathOPi + "Connecting.png")
image_connected = image_prep(imagePathOPi + "Connected.png")
image_feedback = image_prep(imagePathOPi + "Feedback.png")

# Reads off an csv file to create input and result columns for the decision tree
def createTrainingSet():
    pathPC = '/home/user/Documents/opi-setup/buttCushion/csv_files/'
    ##################################################################################################
    combinedDf = pd.read_csv(pathPC + 'dataset_1.csv')
    ##################################################################################################
    # Separate the input features (xDf) and labels (yDf)
    xDf = combinedDf.iloc[:, 0:4]
    yDf = combinedDf.iloc[:, 4]
    print("input columns")
    return xDf, yDf

# uses decision tree to predict user posture
def read_posture():
    runs = 20 # number of predictions to be made
    postureDataSet = np.empty((0,5))
    for i in range(runs):
        postureData = []
        try:
            datapoints = uart_read_array()
            xData = pd.DataFrame(data=[datapoints], columns=['1','2','3','4']) # Send all sensor readings in the list to a dataframe
        except ConnectionError:
            raise ConnectionError
        print(xData)
        prediction = clf.predict(xData)
        postureData = np.append(datapoints, prediction)
        postureDataSet = np.vstack((postureDataSet, postureData))
    return postureDataSet

# reads raw data from buttBrick and reads continuously in case of error 
# raises ConnectionError when bluetooth disconnects
def uart_read_array(): 
    inputList = []
    data_format = r'^(\d+\s){3}\d+$'
    data = uart_service.readline().decode("utf-8").replace("\r\n","")
    while not re.match(data_format, data):
        if not uart_connection or not uart_connection.connected:
            raise ConnectionError("Bluetooth disconnected. Repairing...")

        else:
            print("Error in data format. Rereading....")
            data = uart_service.readline().decode("utf-8").replace("\r\n","")
    data = re.split("\s", data)
    print(data)
    for dPoint in data:
        dPoint = float(dPoint) * 0.0000125885
        inputList.append(dPoint)
    return inputList

# saves a dataframe as a csv file into the device with overall posture and touch detection
def save_csv(dataArray, isGoodPosture, hasTouched):
    global fileLabelCounter
    csvFilePath = '/home/orangepi/opi-setup/csv_files/'
    filename = "dataset_{}".format(fileLabelCounter)
    while os.path.exists(csvFilePath+filename):
        fileLabelCounter += 1
        filename = f'dataset_{fileLabelCounter}.csv'
    print(csvFilePath+filename)
    postureColumn = np.array([[int(isGoodPosture)]] * dataArray.shape[0])
    touchColumn = np.array([[int(hasTouched)]] * dataArray.shape[0])
    dataArray = np.hstack((dataArray, postureColumn))
    dataArray = np.hstack((dataArray, touchColumn))

    # Save the array to a CSV file
    # location, array, delimiter, data format
    np.savetxt(csvFilePath + filename, dataArray, delimiter=",", fmt="%.7f")
    fileLabelCounter += 1
    print("SAVE DONE")

# returns whether the user is sitting on the cushion
def isPresent():
    presenceCheckRuns = 10
    presenceCounter = 0
    threshold = 0.005
    presenceStartTime = time.time()
    for i in range(presenceCheckRuns):
        try:
            presenceCounter += int(threshold < np.mean(uart_read_array()))
        except ConnectionError:
            raise ConnectionError
    presenceTime = time.time() - presenceStartTime
    print(presenceTime)
    if (presenceCounter >= presenceCheckRuns - 5):
        return True
    else:
        return False

# code to be run when the user is sitting on the cushion for the first time after bootup 
def first_run():
    calibration_set = pd.DataFrame(columns=['1','2','3','4','Posture'])
    SampleBatchSize = 5
    sampleCounter = 0
    print("SAVING STARTS HERE")
    start_time = time.time()
    while sampleCounter < SampleBatchSize: # Gets 10 samples of 4-reading samples
        try:
            inputList = uart_read_array()
        except ConnectionError:
            raise ConnectionError
        sampleCounter += 1
        calibration_set.loc[len(calibration_set)] = [inputList[0], inputList[1], inputList[2], inputList[3], 1]
        print(inputList)
        inputList.clear()
    elapsed_time = time.time() - start_time

def calibration():
    disp.image(image_cali_start)
    global isFirstRun
    if (isFirstRun):
        isFirstRun = False
        education_image_loop()          
    isGoodPosture = False
    while not (isGoodPosture):
        try:
            dataArray = read_posture()
        except ConnectionError:
            raise ConnectionError
        goodPostureCount = np.sum(dataArray[:, 4] == 1) # count the occurance of 1
        if (goodPostureCount > len(dataArray) // 2): # more than half of predictions are 1
            break
        else:
            print("fail")

        for i in range(21):
            if (i == 0):
                disp.image(image_cali_fail)
            elif (i == 3):
                disp.image(image_cali_1)
            elif (i == 6):
                disp.image(image_cali_2)
            elif (i == 9):
                disp.image(image_cali_3)
            elif (i == 12):
                disp.image(image_cali_4)
            elif (i == 15):
                disp.image(image_cali_5)
            elif(i == 18):
                disp.image(image_cushion_placement)
            if (touch_pin.value):
                print("force quit")
                return
            sleep(0.5) 
    disp.image(image_cali_success)
    sleep(2)

# prompts the user to check if the predicted posture is correct
def isTouch():
    for i in range (10):
        if (touch_pin.value):
            return True
        sleep (0.5)
    return False
    
# mainframe of code 
# if code reaches this function, it means that the user is sitting on the cushion. 
def run_posture():
    isGoodPosture = False
    hasTouched = False
    inputDf = pd.DataFrame(columns=['1','2','3','4','Posture','Press'])
    try:
        dataArray = read_posture()
    except ConnectionError:
        raise ConnectionError
    goodPostureCount = np.sum(dataArray[:, 4] == 1) # count the occurance of 1
    if (goodPostureCount > len(dataArray) // 2): # more than half of predictions are 1
        isGoodPosture = True
        disp.image(image_good_pos_pred)
        print("good")
    else:
        disp.image(image_bad_pos_pred)
        print("bad")
    
    if (isTouch()):
        hasTouched = True
        disp.image(image_feedback)
        print("touched")
    else:
        print("no touch")
    return dataArray, isGoodPosture, hasTouched

def education_image_loop():
    for i in range(15):
        if (i == 3):
            disp.image(image_cushion_placement)
        elif (i == 6):
            disp.image(image_cali_1)
        elif (i == 9):
            disp.image(image_cali_2)
        elif (i == 12):
            disp.image(image_cali_3)
        elif (i == 15):
            disp.image(image_cali_4)
        elif(i == 18):
            disp.image(image_cali_5)
        sleep(1)

uart_connection = None
ble = BLERadio()
# the main bluetooth connection loop
while True:
    if not uart_connection:
        print("Trying to connect...")
        disp.image(image_connecting)
        for adv in ble.start_scan(ProvideServicesAdvertisement):
            if UARTService in adv.services:
                uart_connection = ble.connect(adv)
                print("Connected")
                break
        ble.stop_scan()

    if uart_connection and uart_connection.connected:
        # bluetooth connected
        uart_service = uart_connection[UARTService]
        disp.image(image_connected)
        while uart_connection.connected:
            try:
                if not (isPresent()): # no user present, run the main loop again
                    disp.image(image_no_presence) 
                    isRisingPresence = False 
                    continue 
                elif not (isRisingPresence):      
                    disp.image(image_yes_presence)
                    sleep(1)
                    calibration()
                    isRisingPresence = True
                disp.image(image_reading)
                dataArray, isGoodPosture, hasTouched = run_posture()
                save_csv(dataArray, isGoodPosture, hasTouched)
            except ConnectionError:
                uart_connection = False
                break 
