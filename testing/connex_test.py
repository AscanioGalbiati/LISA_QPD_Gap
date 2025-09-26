import pyvisa as visa
import time
import numpy as np
import matplotlib.pyplot as plt
import trans_stage as stage
import re


# -------------------- Initialise PyVISA -------------------- #

rm = visa.ResourceManager()

lta00 = rm.open_resource('COM6') 

#lta00.query('*IDN?')

print('\n-------------------- Configuring (LTA) 00 --------------------')
lta00 = stage.conex.connect(rm,
                            "COM6",
                            921600, #9600 for com4
                            1,
                            0.0,
                            "false")

print('\n-------------------- MOVING --------------------')
#lta00.write('1OR') 
zpos = stage.conex.move(lta00, 1, 15.0, 'abs')

lta00.close()




