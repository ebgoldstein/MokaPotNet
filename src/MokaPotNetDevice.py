# MokaPotNetCam - By: ebgoldstein - Wed Jan 26 2022

import pyb, sensor, image, time, os, tf, random


#setup LEDs and set into known off state
redLED = pyb.LED(1) # built-in red LED
greenLED = pyb.LED(2) # built-in green LED
blueLED = pyb.LED(3) # built-in blue LED

#blue light during setup
blueLED.on()

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.skip_frames(time = 2000)


#Load the TFlite model and the labels
net = tf.load('/MokaPotNet_cat.tflite', load_to_fb=True)
#labels = ['No MokaPot', 'MokaPot']

#turn blue off when model is loaded
blueLED.off()


#MAIN LOOP



while(True):

    #toggle LED for visual debugging that script is running
    #blueLED.toggle()

    #get the image/take the picture
    img = sensor.snapshot()

    #Do the classification and get the object returned by the inference.
    TF_objs = net.classify(img)
    #print(TF_objs)

    #The object has a output, which is a list of classifcation scores
    #for each of the output channels. this model only has 2 (no mokapot, mokapot).
    NoPot = TF_objs[0].output()[0]
    Pot = TF_objs[0].output()[1]

    print(Pot)
    print(NoPot)
    if Pot > NoPot:
     greenLED.on()
     pyb.delay(100)
     greenLED.off()
    else:
     redLED.on()
     pyb.delay(100)
     redLED.off()

    #add a delay, if you want to wait some number of milliseconds
    #pyb.delay(1000)
