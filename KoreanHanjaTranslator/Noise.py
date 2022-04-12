# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np
from PIL import Image
import PIL.ImageOps
import random

# Creating images for copies of TwoStrokes.png
twoStrokes = Image.open('Images/TwoStrokes.png')
twoStrokes_array = np.array(twoStrokes)
height = twoStrokes.size[0]
width = twoStrokes.size[1]

# Creating copies
for z in range(25):
    # Mixing Pixels
    for x in range(int(height / 50)):
        for y in range(int(width / 100)):
            left = twoStrokes.getpixel((x, y))
            right = twoStrokes.getpixel((x+1, y))
            twoStrokes.putpixel((x+1, y), left)
            twoStrokes.putpixel((x, y), right)
    i = random.randrange(1, 100)
    if i > 50:
        twoStrokesInverted = twoStrokes.resize((random.randrange(150, 250), random.randrange(150, 250)))
        twoStrokesInverted.save("Images/TwoStrokes" + str(z+1) + ".png")
    if i < 50:
        twoStrokes = PIL.ImageOps.invert(twoStrokes)
        twoStrokes.save("Images/TwoStrokes" + str(z+1) + ".png")

# Creating images for copies of SixStrokes.png
sixStrokes = Image.open('Images/SixStrokes.png')
sixStrokes_array = np.array(sixStrokes)
height = sixStrokes.size[0]
width = sixStrokes.size[1]

for z in range(25):
    for x in range(int(height / 50)):
        for y in range(int(width / 100)):
            left = sixStrokes.getpixel((x, y))
            right = sixStrokes.getpixel((x+1, y))
            sixStrokes.putpixel((x+1, y), left)
            sixStrokes.putpixel((x, y), right)
    i = random.randrange(1, 100)
    if i > 50:
        sixStrokesInverted = sixStrokes.resize((random.randrange(150, 250), random.randrange(150, 250)))
        sixStrokesInverted.save("Images/SixStrokes" + str(z + 1) + ".png")
    if i < 50:
        sixStrokes = PIL.ImageOps.invert(sixStrokes)
        sixStrokes.save("Images/SixStrokes" + str(z+1) + ".png")

# Creating images for copies of SevenStrokes.png
SevenStrokes = Image.open('Images/SevenStrokes.png')
SevenStrokes_array = np.array(SevenStrokes)
height = SevenStrokes.size[0]
width = SevenStrokes.size[1]

for z in range(25):
    for x in range(int(height / 50)):
        for y in range(int(width / 100)):
            left = SevenStrokes.getpixel((x, y))
            right = SevenStrokes.getpixel((x+1, y))
            SevenStrokes.putpixel((x+1, y), left)
            SevenStrokes.putpixel((x, y), right)
    i = random.randrange(1, 100)
    if i > 50:
        SevenStrokesInverted = SevenStrokes.resize((random.randrange(150, 250), random.randrange(150, 250)))
        SevenStrokesInverted.save("Images/SevenStrokes" + str(z + 1) + ".png")
    if i < 50:
        SevenStrokes = PIL.ImageOps.invert(SevenStrokes)
        SevenStrokes.save("Images/SevenStrokes" + str(z+1) + ".png")

# Creating images for copies of TenStrokes.png
tenStrokes = Image.open('Images/TenStrokes.png')
tenStrokes_array = np.array(tenStrokes)
height = tenStrokes.size[0]
width = tenStrokes.size[1]

for z in range(25):
    for x in range(int(height / 50)):
        for y in range(int(width / 100)):
            left = tenStrokes.getpixel((x, y))
            right = tenStrokes.getpixel((x+1, y))
            tenStrokes.putpixel((x+1, y), left)
            tenStrokes.putpixel((x, y), right)
    i = random.randrange(1, 100)
    if i > 50:
        tenStrokesInverted = tenStrokes.resize((random.randrange(150, 250), random.randrange(150, 250)))
        tenStrokesInverted.save("Images/TenStrokes" + str(z + 1) + ".png")
    if i < 50:
        tenStrokes = PIL.ImageOps.invert(tenStrokes)
        tenStrokes.save("Images/TenStrokes" + str(z+1) + ".png")

# Creating images for copies of ThirteenStrokes.png
thirteenStrokes = Image.open('Images/ThirteenStrokes.png')
thirteenStrokes_array = np.array(thirteenStrokes)
height = thirteenStrokes.size[0]
width = thirteenStrokes.size[1]

for z in range(25):
    for x in range(int(height / 50)):
        for y in range(int(width / 100)):
            left = thirteenStrokes.getpixel((x, y))
            right = thirteenStrokes.getpixel((x+1, y))
            thirteenStrokes.putpixel((x+1, y), left)
            thirteenStrokes.putpixel((x, y), right)
    i = random.randrange(1, 100)
    if i > 50:
        thirteenStrokesInverted = thirteenStrokes.resize((random.randrange(150, 250), random.randrange(150, 250)))
        thirteenStrokesInverted.save("Images/ThirteenStrokes" + str(z + 1) + ".png")
    if i < 50:
        thirteenStrokes = PIL.ImageOps.invert(thirteenStrokes)
        thirteenStrokes.save("Images/ThirteenStrokes" + str(z+1) + ".png")

# Creating images for copies of FourteenStrokes.png
fourteenStrokes = Image.open('Images/FourteenStrokes.png')
fourteenStrokes_array = np.array(fourteenStrokes)
height = fourteenStrokes.size[0]
width = fourteenStrokes.size[1]

for z in range(25):
    for x in range(int(height / 50)):
        for y in range(int(width / 100)):
            left = fourteenStrokes.getpixel((x, y))
            right = fourteenStrokes.getpixel((x+1, y))
            fourteenStrokes.putpixel((x+1, y), left)
            fourteenStrokes.putpixel((x, y), right)
    i = random.randrange(1, 100)
    if i > 50:
        fourteenStrokesInverted = fourteenStrokes.resize((random.randrange(150, 250), random.randrange(150, 250)))
        fourteenStrokesInverted.save("Images/FourteenStrokes" + str(z + 1) + ".png")
    if i < 50:
        fourteenStrokes = PIL.ImageOps.invert(fourteenStrokes)
        fourteenStrokes.save("Images/FourteenStrokes" + str(z+1) + ".png")