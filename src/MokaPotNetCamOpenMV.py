# MokaPotNetCam - By: ebgoldstein - Wed Jan 26 2022

import sensor, image, time, lcd, pyb, tf

#setup LEDs and set into known off state
redLED   = pyb.LED(1)
greenLED = pyb.LED(2)
yellowLED  = pyb.LED(3)

sensor.reset() # Initialize the camera sensor.
sensor.set_pixformat(sensor.RGB565) # or sensor.GRAYSCALE
sensor.set_framesize(sensor.QQVGA2) # Special 128x160 framesize for LCD Shield.
sensor.skip_frames(time = 2000)
lcd.init() # Initialize the lcd screen.

#blue light during setup
yellowLED.on()



#Load the TFlite model and the labels
net = tf.load('/MokaPotNet_cat_OMV.tflite', load_to_fb=True)
labels = ['No MokaPot', 'MokaPot']

#turn blue off when model is loaded
yellowLED.off()


#MAIN LOOP

while(True):

    img = sensor.snapshot()

    #Do the classification and get the object returned by the inference.
    TF_objs = net.classify(img)
    #print(TF_objs)

    #The object has a output, which is a list of classifcation scores
    #for each of the output channels. this model only has 2 (no mokapot, mokapot).
    NoPot = TF_objs[0].output()[0]
    Pot = TF_objs[0].output()[1]

    if Pot > NoPot:
     img.draw_string(1,140, "No Moka Pot", color = (10,10,100), scale = 2,mono_space = False)
    else:
     img.draw_string(1,140, "Moka Pot", color = (10,10,100), scale = 2,mono_space = False)



    lcd.display(img) # dsiplay image
