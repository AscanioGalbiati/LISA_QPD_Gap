import io
import os
import time
import json
import pathlib
import pickle
import datetime
import numpy as np
import pyvisa as visa
import matplotlib.pyplot as plt
import pyvisa_util as pvu
import trans_stage as stage
import switchbox_rf as srf
import multimeter as dmm
import serial

# ============================================================ #
#                       Hard coded paths                       #
# ============================================================ #

# Script root path
root_dir = pathlib.Path(__file__).resolve().parent

# Config JSON path
json_config_path = '../ascanio/config'
json_config_fname = 'focus_laser_config_phasespace_xy_asca_manual.json'
json_config_fullpath = pathlib.Path((root_dir / json_config_path).resolve(), json_config_fname)

# Data save root path
data_save_root = pathlib.Path('C:/Users/virgorun/Projects/LISA/daq/experiments/Students/ascanio/data')

# ============================================================ #
#                       Custom functions                       #
# ============================================================ #

def datetoday(strfmt):
    today = datetime.date.today()
    return today.strftime(strfmt)

def timenow(strfmt):
    tn = time.localtime()
    return time.strftime(strfmt, tn)

def getDuration(start, end, strfmt=None):
    '''
    Calculate the duration between two date and times
    '''
    # Use the datetime object to convert and calculate the time difference
    detla_time = datetime.datetime.strptime(end, strfmt) - datetime.datetime.strptime(start, strfmt)
    
    delta_totalseconds = detla_time.total_seconds()
    delta_year, remain_year = divmod(delta_totalseconds, 31536000)
    delta_days, remain_days = divmod(remain_year, 86400)
    delta_hours, remain_hours = divmod(remain_days, 3600)
    delta_minutes, remain_minutes = divmod(remain_hours, 60)
    
    print('Duration: {:02d} Years, {:02d} Days, {:02d} Hours, {:02d} Minutes, {:02d} Seconds\n'.format(int(delta_years),
                                                                                                 int(delta_days),
                                                                                                 int(delta_hours),
                                                                                                 int(delta_minutes),
                                                                                                 int(remain_minutes)))
    
    return int(delta_year), int(delta_days), int(delta_hours), int(delta_minutes), int(remain_minutes)

def calcDuration(lenx, leny, navg=1, *args):
    '''
    Estimate the duration of the script
    '''
    # Start with estimated time of the translation stage move time (assuming 5 seconds)
    est_time_sec = lenx*leny*10
    
    for i in args:
        est_time_sec += (lenx*leny*i*navg)
    delta_year, remain_year = divmod(est_time_sec, 31536000)
    delta_days, remain_days = divmod(remain_year, 86400)
    delta_hours, remain_hours = divmod(remain_days, 3600)
    delta_minutes, Remain_minutes = divmod(remain_hours, 60)
    
    print('Duration: {:02d} Years, {:02d} Days, {:02d} Hours, {:02d} Minutes, {:02d} Seconds\n'.format(int(delta_year),
                                                                                                 int(delta_days),
                                                                                                 int(delta_hours),
                                                                                                 int(delta_minutes),
                                                                                                 int(remain_minutes)))
    
    return int(delta_year), int(delta_days), int(delta_hours), int(delta_minutes), int(remain_minutes)

def erf_fit(x,a,b,c,d):
    #return a*speci.erfc(b*(x-c))+d
    return a*speci.erf(b*(x-c))+d

def erfc_fit(x,a,b,c,d):
    return 1 - erf_fit(x,a,b,c,d)

def beam_waist(x):
    return (1/(x*np.sqrt(2)))
    #return np.sqrt(2)/x

def dbm2w(x, rbw):
    #dbm_Hz = x - 10.0*np.log10(rbw)
    return 1.0 * (10.0 **(x/10.0))/1E3

def mfunc(x, a, b, c):
    '''
    Define the M Squared function
    '''
    return a * np.sqrt(1 + ((x-c)**2 * 1056E-9**2 * b**4)/(np.pi**2 * a**4))

def mfunc2(x, a, c):
    '''
    Define the M Squared function with fixed M^2
    '''
    return a * np.sqrt(1 + ((x-c)**2 * 1056E-9**2 * 1.0**4)/(np.pi**2 * a**4))

# ============================================================ #
#                       Config experiment                      #
# ============================================================ #

# -------------------- Load config -------------------- #
with open(json_config_fullpath, 'r') as fopen:
    focuslaser_jsonconfig = json.load(fopen)
    
# -------------------- Initialise PyVISA -------------------- #
rm = visa.ResourceManager()

# -------------------- DMM config -------------------- #
if 'dmm00' in focuslaser_jsonconfig.keys():
    
    # Connect to the arbitrary waveform generator
    dmm00 = rm.open_resource(focuslaser_jsonconfig['dmm00']['conn']['address'], access_mode=0, send_end=True)

    # Reset the instrument
    dmm00.write("*RST")

    # Send the list of commands to configure the instrument
    print('\n-------------------- Configuring Digital Multimeter (DMM) 00 --------------------')
    pvu.pyvisa_cmd_list(dmm00, focuslaser_jsonconfig['dmm00']['cmd'], 'w')
    dmm00.write("CONF:CURR:DC")  # Set to current measurement

    # Voltage test measurement
    voltage = float(dmm00.query("READ?"))
    print(f"Measured Voltage: {voltage:.2f} V")
    
    # Current test measurement
    current = float(dmm00.query("READ?"))
    print(f"Measured Photocurrent: {current:.7f} A")

# -------------------- Switch Matrix config -------------------- #
# Serial connection for the switch matrix
ser = serial.Serial('COM5', 19200, timeout=1)

# Address the instrument at GPIB 7
ser.write(b'++addr7\r')

# Disable specific status queue events
ser.write(b':stat:que:dis (-410,-410)\n')
time.sleep(0.1)

# Functions to switch connections for OUT1 
def connect_quadD_out1(ser):  # IN3 to OUT1 for Quadrant D
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!39)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!40)\n')
    time.sleep(0.1)

def connect_quadA_out1(ser):  # IN4 to OUT1 for Quadrant A
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!38)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!39)\n')
    time.sleep(0.1)

print("Trying to open LTA00 stage on:", focuslaser_jsonconfig['lta00']['conn']['address'])
if 'lta00' in focuslaser_jsonconfig.keys():
    print('\n-------------------- Configuring Y Stage (LTA) 00 --------------------')
    lta00 = stage.conex.connect(rm,
                                focuslaser_jsonconfig['lta00']['conn']['address'],
                                focuslaser_jsonconfig['lta00']['conn']['baud_rate'],
                                focuslaser_jsonconfig['lta00']['conn']['channel'],
                                focuslaser_jsonconfig['lta00']['conn']['query_delay'],
                                focuslaser_jsonconfig['lta00']['conn']['force'])
    # Force home movement. Comment out line if not want to go home. TODO, fix COEX homing.
    lta00.write('1OR') 
    ypos = stage.conex.move(lta00, 1, float(focuslaser_jsonconfig['global_params']['ystart_mm']), 'abs')

# -------------------- TRAC stage config -------------------- #
print("Trying to open TRAC00 stage on:", focuslaser_jsonconfig['lta01']['conn']['address'])
if 'lta01' in focuslaser_jsonconfig.keys():
    print('\n-------------------- Configuring X Stage (TRAC) 00 --------------------')
    lta01 = stage.conex.connect(rm,
                              focuslaser_jsonconfig['lta01']['conn']['address'],
                              focuslaser_jsonconfig['lta01']['conn']['baud_rate'],
                              focuslaser_jsonconfig['lta01']['conn']['channel'],
                              focuslaser_jsonconfig['lta01']['conn']['query_delay'],
                              focuslaser_jsonconfig['lta01']['conn']['force'])
    # Force home movement. Comment out line if not want to go home. TODO, fix COEX homing.
    lta01.write('1OR') 
    xpos = stage.conex.move(lta01, 1, float(focuslaser_jsonconfig['global_params']['xstart_mm']), 'abs')

# -------------------- Configure scan parameters -------------------- #

# Calculate the step in x direction
step_x = int((focuslaser_jsonconfig['global_params']['xstop_mm'] - focuslaser_jsonconfig['global_params']['xstart_mm'])/focuslaser_jsonconfig['global_params']['xstep_mm']) + 1

 # Generate x array for the scan
x_array = np.linspace(focuslaser_jsonconfig['global_params']['xstart_mm'], focuslaser_jsonconfig['global_params']['xstop_mm'], step_x)

# Calculate the step in y direction
step_y = int((focuslaser_jsonconfig['global_params']['ystop_mm'] - focuslaser_jsonconfig['global_params']['ystart_mm'])/focuslaser_jsonconfig['global_params']['ystep_mm']) + 1

# Generate y array for the scan
y_array = np.linspace(focuslaser_jsonconfig['global_params']['ystart_mm'], focuslaser_jsonconfig['global_params']['ystop_mm'], step_y)

# -------------------- Save location config -------------------- #
# Data save location
save_dir_base = pathlib.Path('{}/{}/GapScan/{}_{}_{}_{}_{}_{}_250924_LB1471C_quadA&D_manual_setup_Z50mm_Yscan_gap_identification_precise'.format(
    data_save_root, 
    focuslaser_jsonconfig['dut']['batch'],
    focuslaser_jsonconfig['dut']['foundry'],
    focuslaser_jsonconfig['dut']['wafer'],
    focuslaser_jsonconfig['dut']['dut_type'],
    focuslaser_jsonconfig['dut']['size'],
    focuslaser_jsonconfig['dut']['gap'],
    focuslaser_jsonconfig['dut']['dut_id']
))

# Check the save path and make a folder if it doesn't exist.
if not save_dir_base.exists():
    save_dir_base.mkdir(parents=True, exist_ok=True)
    
save_fname_str = "{}_{}_{}_{}_{}_{}_{}_QPDGapScan_KnifeEdge_250924_LB1471C_quadA&D_manual_setup_Z50mm_Yscan_gap_identification_precise".format(
    focuslaser_jsonconfig['dut']['batch'],
    focuslaser_jsonconfig['dut']['foundry'],
    focuslaser_jsonconfig['dut']['wafer'],
    focuslaser_jsonconfig['dut']['dut_type'],
    focuslaser_jsonconfig['dut']['gap'],
    focuslaser_jsonconfig['dut']['gap'],
    focuslaser_jsonconfig['dut']['dut_id']
)

# Define the file extension
fext = '.pkl'

print("Base save directory: {:>100}".format(save_dir_base.as_posix()))                                  
print("Save filename: {:>78}".format(save_fname_str))
print('\n-------------------- Setup Complete! --------------------\n')

# ============================================================ #
#                         Main experiment                      #
# ============================================================ #

# Insert Z position (in micrometers)
z_position_um = 5000

for y in y_array:
    # Move Y stage
    try:
        ypos = stage.conex.move(lta00, 1, float(y), 'abs')
    except visa.VisaIOError as e:
        print(f"Error moving Y stage to {y} mm: {e}")
        lta00.clear()
        time.sleep(1)
        continue
    
    # Create subfolder for this Y position
    save_dir_str = save_dir_base / f"Y{y*1000:04.0f}um"
    if not save_dir_str.exists():
        save_dir_str.mkdir(parents=True, exist_ok=True)
    
    # Configure data storage
    data = {
        'rawdata': {
            'comp_gps_time': np.zeros_like(x_array),
            'stage_laser_xposition': np.zeros_like(x_array),
            'stage_laser_yposition': np.zeros_like(x_array),
            'quadA': {
                'dmm00_curr_amp': np.zeros((len(x_array), focuslaser_jsonconfig['global_params']['number_averages']))
            },
            'quadD': {
                'dmm00_curr_amp': np.zeros((len(x_array), focuslaser_jsonconfig['global_params']['number_averages']))
            }
        }
    }
    
    data['rawdata']['start_date'] = datetoday('%Y%m%d')
    data['rawdata']['start_time'] = timenow('%H%M%S')
    
    for idx, x in enumerate(x_array):
        # Move X stage
        try:
            xpos = stage.conex.move(lta01, 1, float(x), 'abs')
        except visa.VisaIOError as e:
            print(f"Error moving X stage to {x} mm: {e}")
            lta01.clear()
            time.sleep(1)
            continue
            
        # Update the x position
        data['rawdata']['stage_laser_xposition'][idx] = xpos

        # Update the y position
        data['rawdata']['stage_laser_yposition'][idx] = ypos
        
        navg = focuslaser_jsonconfig['global_params']['number_averages']
        
        # Measure DC photocurrent for Quadrant D (IN3 to OUT1)
        try:
            print(f"\nSwitching matrix to Quadrant D (IN3 to OUT1)...")
            connect_quadD_out1(ser)
            time.sleep(1)  # Allow relays to settle
            
            temp_measurements_c = np.zeros(navg)
            for i in range(navg):
                try:
                    dmm00.write("INIT")
                    time.sleep(0.05)
                    temp_measurements_c[i] = dmm00.query_ascii_values("FETC?")[0]
                except Exception as e:
                    print(f"Error in single measurement {i+1} at X={x} mm, Y={y} mm for Quadrant D: {e}")
                    temp_measurements_c[i] = np.nan
            data['rawdata']['quadD']['dmm00_curr_amp'][idx] = temp_measurements_c
            print(f"Y={y} mm, X={x} mm, DC photocurrent Quadrant D = {np.mean(temp_measurements_c):.7f} A")
        except Exception as e:
            print(f"Error measuring at Y={y} mm, X={x} mm for Quadrant D: {e}")
            data['rawdata']['quadD']['dmm00_curr_amp'][idx] = np.nan
        
        # Measure DC photocurrent for Quadrant A (IN4 to OUT1)
        try:
            print(f"\nSwitching matrix to Quadrant A (IN4 to OUT1)...")
            connect_quadA_out1(ser)
            time.sleep(1)  # Allow relays to settle
            
            temp_measurements_a = np.zeros(navg)
            for i in range(navg):
                try:
                    dmm00.write("INIT")
                    time.sleep(0.05)
                    temp_measurements_a[i] = dmm00.query_ascii_values("FETC?")[0]
                except Exception as e:
                    print(f"Error in single measurement {i+1} at X={x} mm, Y={y} mm for Quadrant A: {e}")
                    temp_measurements_a[i] = np.nan
            data['rawdata']['quadA']['dmm00_curr_amp'][idx] = temp_measurements_a
            print(f"Y={y} mm, X={x} mm, DC photocurrent Quadrant A = {np.mean(temp_measurements_a):.7f} A")
        except Exception as e:
            print(f"Error measuring at Y={y} mm, X={x} mm for Quadrant A: {e}")
            data['rawdata']['quadA']['dmm00_curr_amp'][idx] = np.nan

    print(f"DC photocurrent Quadrant D = {np.mean(temp_measurements_c)}")
    print(f"DC photocurrent Quadrant A = {np.mean(temp_measurements_a)}")

    # Save data
    fout = '{}_{}_{}_Y{:04d}um_Z{:05d}um'.format(
        data['rawdata']['start_date'],
        data['rawdata']['start_time'],
        save_fname_str,
        int(np.round(y, decimals=3) * 1000),
        int(np.round(z_position_um, decimals=3))
    )

    # Combine dictionaries
    outdict = focuslaser_jsonconfig | data

    # Save the instrument data 
    try:
        with open(pathlib.Path(save_dir_str, fout).with_suffix(fext), 'wb') as fopen:
            pickle.dump(outdict, fopen)
            print(f"Saving data...\nLocation: {save_dir_str}\nFile Name: {fout}")
    except Exception as e:
        print(f"Error saving data: {e}")
        with open(pathlib.Path(fout).with_suffix(fext), 'wb') as fopen:
            pickle.dump(outdict, fopen)
            print("Saving data...\nLocation: {}\nFile Name: {}".format(os.getcwd(), fout))
    
    # Plot data
    plt.plot(x_array, data['rawdata']['quadD']['dmm00_curr_amp'].mean(axis=1), label='quadD')
    plt.plot(x_array, data['rawdata']['quadA']['dmm00_curr_amp'].mean(axis=1), label='quadA')
    plt.legend()
    plt.title(f'DC Photocurrent at Y={y} mm')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Photocurrent (A)')
    plt.show()

# -------------------- Cleanup -------------------- #
stage.conex.move(lta01, 1, float(focuslaser_jsonconfig['global_params']['xstart_mm']), 'abs')
stage.conex.move(lta00, 1, float(focuslaser_jsonconfig['global_params']['ystart_mm']), 'abs')
ser.write(b':open all\n')
time.sleep(0.5)
ser.close()
rm.close()

print("Experiment complete!")