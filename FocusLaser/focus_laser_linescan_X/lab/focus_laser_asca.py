import io
import os
import time
import json
import pathlib
import pickle
import datetime
import numpy as np
import scipy.signal as spsig
import scipy.special as speci
import pyvisa as visa
import matplotlib.pyplot as plt
import pyvisa_util as pvu
import trans_stage as stage
import switchbox_rf as srf
import multimeter as dmm
import sourcemeter as smu
import spectrum_analyser as spa
from scipy.optimize import curve_fit

# ============================================================ #
#                       Hard coded paths                       #
# ============================================================ #

# Script root path
root_dir = pathlib.Path(__file__).resolve().parent

# Config JSON path
json_config_path = '../config/FocusLaser'
json_config_fname = 'focus_laser_config_asca.json'
json_config_fullpath = pathlib.Path((root_dir / json_config_path).resolve(), json_config_fname)

# Data save root path
data_save_root = pathlib.Path('C:/Users/virgorun/Projects/LISA/data')

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
    delta_minutes, remain_minutes = divmod(remain_hours, 60)

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
'''
try:
    stage_conn = rm.open_resource("COM6", baud_rate=912600)  # Adjust baud rate as needed
    print("Connected successfully:", stage_conn.query("*IDN?"))  # Or another ID command
    stage_conn.close()
except visa.VisaIOError as e:
    print(f"Error connecting to COM6: {e}")'''

# -------------------- DMM config -------------------- #

if 'dmm00' in focuslaser_jsonconfig.keys():
    
    # Connect to the arbitrary waveform generator
    dmm00 = rm.open_resource(focuslaser_jsonconfig['dmm00']['conn']['address'], access_mode=0, send_end=True)

    # Reset the instrument
    dmm00.write("*RST")
    
    # Send the list of commands to configure the instrument
    print('\n-------------------- Configuring Digital Multimeter (DMM) 00 --------------------')
    pvu.pyvisa_cmd_list(dmm00, focuslaser_jsonconfig['dmm00']['cmd'], 'w')

    # Voltage test measurement
    dmm00.write("CONF:VOLT:DC")
    
    voltage = float(dmm00.query("READ?"))
    print(f"Measured Voltage: {voltage:.2f} V")

    # Current test measurement
    dmm00.write("CONF:CURR:DC")
    
    current = float(dmm00.query("READ?"))
    print(f"Measured Photocurrent: {current:.7f} A")

# -------------------- Z stage config -------------------- #
if 'lta00' in focuslaser_jsonconfig.keys():

    print('\n-------------------- Configuring (LTA) 00 --------------------')
    #Connect to the NSC stage for XY translation
    lta00 = stage.conex.connect(rm,
                              focuslaser_jsonconfig['lta00']['conn']['address'],
                              focuslaser_jsonconfig['lta00']['conn']['baud_rate'],
                              focuslaser_jsonconfig['lta00']['conn']['channel'],
                              focuslaser_jsonconfig['lta00']['conn']['query_delay'],
                              focuslaser_jsonconfig['lta00']['conn']['force']
                               )

    # Force home movement. Comment out line if not want to go home. TODO, fix COEX homing.
    lta00.write('1OR') 
    
    zpos = stage.conex.move(lta00, 1, float(focuslaser_jsonconfig['global_params']['zstart_um']), 'abs')


# -------------------- NSC stage config -------------------- #
if 'nsc00' in focuslaser_jsonconfig.keys():

    print('\n-------------------- Configuring Newport NewStep Translation Stages (NSC) 00 --------------------')
    
    #Connect to the NSC stage for XY translation
    nsc00 = stage.nsc.connect(rm,
                              focuslaser_jsonconfig['nsc00']['conn']['address'],
                              focuslaser_jsonconfig['nsc00']['conn']['baud_rate'],
                              focuslaser_jsonconfig['nsc00']['conn']['query_delay'],
                              focuslaser_jsonconfig['nsc00']['conn']['timeout'])
    print(stage.nsc)
    print(dir(stage.nsc))
    
    try:
        #stage.nsc.move(nsc00, 2, 10000, 1.0)
        #stage.nsc.move(nsc00, 2, 'home')
        #stage.nsc.move(nsc00, 7, 10000, 1.0)
        stage.nsc.move(nsc00, 7, 'home')
    except AttributeError:
        print("Error: 'nsc_move' method not found.")
    
    # Move stages to starting position
    xpos = stage.nsc.move(nsc00, 7, focuslaser_jsonconfig['global_params']['xstart_um'])
    ypos = stage.nsc.move(nsc00, 2, focuslaser_jsonconfig['global_params']['ystart_um'])

    # Home the stages
    #stage.nsc.nsc_move(nsc00, 1, 'home')
    #stage.nsc.nsc_move(nsc00, 2, 'home')
    #stage.nsc.nsc_move(nsc00, 7, 'home')
    #stage.nsc.nsc_move(nsc00, 8, 'home')

    # Move stages to starting position
    #xpos = stage.nsc.nsc_move(nsc00, 7, focuslaser_jsonconfig['global_params']['xstart_um'])
    #ypos = stage.nsc.nsc_move(nsc00, 8, focuslaser_jsonconfig['global_params']['ystart_um'])


# Configure array for line scan

if focuslaser_jsonconfig['global_params']['xstart_um'] == focuslaser_jsonconfig['global_params']['xstop_um']:
    print('Vertical (Y) line scan')

    # Calculate the steps
    step = int((focuslaser_jsonconfig['global_params']['ystop_um'] - focuslaser_jsonconfig['global_params']['ystart_um'])/focuslaser_jsonconfig['global_params']['xystep_um']) + 1

    # Generate arrays for the scan
    y_array = np.linspace(focuslaser_jsonconfig['global_params']['ystart_um'], focuslaser_jsonconfig['global_params']['ystop_um'], step)

elif focuslaser_jsonconfig['global_params']['ystart_um'] == focuslaser_jsonconfig['global_params']['ystop_um']:
    print('Horizontal (X) line scan')

    # Calculate the steps
    step = int((focuslaser_jsonconfig['global_params']['xstop_um'] - focuslaser_jsonconfig['global_params']['xstart_um'])/focuslaser_jsonconfig['global_params']['xystep_um']) + 1

    # Generate arrays for the scan
    x_array = np.linspace(focuslaser_jsonconfig['global_params']['xstart_um'], focuslaser_jsonconfig['global_params']['xstop_um'], step)
        
# -------------------- Save location config -------------------- #
# Data save location
save_dir_str = pathlib.Path('{}/{}/GapScan/{}_{}_{}_{}_{}_{}_asca_8'.format(data_save_root, 
                                                                     focuslaser_jsonconfig['dut']['batch'],
                                                                     focuslaser_jsonconfig['dut']['foundry'],
                                                                     focuslaser_jsonconfig['dut']['wafer'],
                                                                     focuslaser_jsonconfig['dut']['dut_type'],
                                                                     focuslaser_jsonconfig['dut']['size'],
                                                                     focuslaser_jsonconfig['dut']['gap'],
                                                                     focuslaser_jsonconfig['dut']['dut_id']))
# Check the save path and make a folder if it doesn't exist.
if not save_dir_str.exists():
    save_dir_str.mkdir(parents=True, exist_ok=True)
    
save_fname_str = "{}_{}_{}_{}_{}_{}_{}_QPDGapScan_KnifeEdge_asca_8".format(focuslaser_jsonconfig['dut']['batch'],
                                                                  focuslaser_jsonconfig['dut']['foundry'],
                                                                  focuslaser_jsonconfig['dut']['wafer'],
                                                                  focuslaser_jsonconfig['dut']['dut_type'],
                                                                  focuslaser_jsonconfig['dut']['gap'],
                                                                  focuslaser_jsonconfig['dut']['gap'],
                                                                  focuslaser_jsonconfig['dut']['dut_id'])

# Define the file extension
fext = '.pkl'

#print('-----------------------------------------------------------------------------------------------------------------------')
print("Save directory: {:>100}".format(save_dir_str.as_posix()))                                  
print("Save filename: {:>78}".format(save_fname_str))
#print('-----------------------------------------------------------------------------------------------------------------------')
print('\n-------------------- Setup Complete! --------------------\n')

''' DELETE
# -------------------- Configure scan parameters -------------------- #
xstart = focuslaser_jsonconfig['global_params']['xstart_um']
xstop = focuslaser_jsonconfig['global_params']['xstop_um']
xstep = focuslaser_jsonconfig['global_params']['xystep_major_um']
ystart = focuslaser_jsonconfig['global_params']['ystart_um']
zstart = focuslaser_jsonconfig['global_params']['zstart_um']
zstop = focuslaser_jsonconfig['global_params']['zstop_um']
zstep = 0.03  # Define Z step size in micrometers (adjust as needed)
navg = focuslaser_jsonconfig['global_params']['number_averages']

# Generate arrays
x_array = np.arange(xstart, xstop + xstep, xstep)
z_array = np.arange(zstart, zstop + zstep, zstep)

print(f"Horizontal scan from X={xstart} to {xstop} um, step={xstep} um")
print(f"Z scan from {zstart} to {zstop} um, step={zstep} um")
print(f"Y fixed at {ystart} um")
print(f"Number of averages: {navg}")
'''

# ============================================================ #
#                         Main experiment                      #
# ============================================================ #

# -------------------- Configure data storage -------------------- #
# Make empty arrays for data storage 
'''data = {'rawdata': {
                    'comp_gps_time' : np.zeros_like(x_array),
                    'smu00_curr_amp' : np.zeros((len(x_array), focuslaser_jsonconfig['global_params']['number_averages'])),
                    'smu00_bias_volt' : np.zeros((len(x_array), focuslaser_jsonconfig['global_params']['number_averages'])),
                    'th2e01_temp_degreec' : np.zeros_like(x_array),
                    'th2e01_humd_percent' : np.zeros_like(x_array),
                    'th2e01_dewp_degreec' : np.zeros_like(x_array),
                    'stage_laser_xposition' : np.zeros_like(x_array),
                    #'stage_laser_xoffset' : Xmain_pos,
                    'stage_laser_yposition' : np.zeros_like(x_array),
                    'stage_laser_zposition' : np.array([]),
                    }
       }'''

data = {
    'rawdata': {
        'comp_gps_time': np.zeros_like(x_array),
        'stage_laser_xposition': np.zeros_like(x_array),
        'stage_laser_yposition': np.zeros_like(x_array),
        'stage_laser_zposition': np.array([]),
        'quadD': {
            'dmm00_curr_amp': np.zeros((len(x_array), focuslaser_jsonconfig['global_params']['number_averages']))
        }
    }
}

zstep = focuslaser_jsonconfig['global_params']['zstep_um']
while zpos > focuslaser_jsonconfig['global_params']['zstop_um']:

    data['rawdata']['start_date'] = datetoday('%Y%m%d')
    data['rawdata']['start_time'] = timenow('%H%M%S')

    # Move Z stage to current position
    stage.conex.move(lta00, 1, float(zpos), 'abs')
    data['rawdata']['stage_laser_zposition'] = np.append(data['rawdata']['stage_laser_zposition'], zpos)

    for idx, x in enumerate(x_array):
        # Move X stage
        try:
            xpos = stage.nsc.move(nsc00, 7, x)
        except:
            xpos = stage.nsc.move(nsc00, 1, float(x))

        # Update the x positoin
        data['rawdata']['stage_laser_xposition'][idx] = xpos

        # Ensure Y stage is at correct position. If not, then move it to the correct place.
        if ypos != focuslaser_jsonconfig['global_params']['ystart_um']:
            try:
                ypos = stage.nsc.nsc_move(nsc00, 2, focuslaser_jsonconfig['global_params']['ystart_um'])
            except:
                ypos = stage.nsc.nsc_move(nsc00, 2, float(focuslaser_jsonconfig['global_params']['ystart_um']))
        # Update the y position
        data['rawdata']['stage_laser_yposition'][idx] = ypos
        '''
        # Measure DC photocurrent for Quadrant D
        data['rawdata']['quadD']['dmm00_curr_amp'][idx] = dmm.key34.multi_meas(
            dmm00, focuslaser_jsonconfig['global_params']['number_averages']
        )
        '''
        # Measure DC photocurrent for Quadrant D with 20 single measurements
        try:
            number_averages = focuslaser_jsonconfig['global_params']['number_averages']
            temp_measurements = np.zeros(number_averages)
            for i in range(number_averages):
                try:
                    dmm00.write("INIT")
                    time.sleep(0.05)  # Wait for DMM to be ready
                    temp_measurements[i] = dmm00.query_ascii_values("FETC?")[0]
                except Exception as e:
                    print(f"Error in single measurement {i+1} at X={x} um, Z={zpos} um: {e}")
                    temp_measurements[i] = np.nan
            data['rawdata']['quadD']['dmm00_curr_amp'][idx] = temp_measurements
        except Exception as e:
            print(f"Error measuring photocurrent at X={x} um, Z={zpos} um: {e}")
            data['rawdata']['quadD']['dmm00_curr_amp'][idx] = np.nan

        print(f"DC photocurrent = {np.mean(temp_measurements)}")
        '''
        # Measure DC photocurrent for Quadrant D with 20 single measurements
        try:
            # Ensure RF switch is set to Quadrant D
            srf.mcsp4t.quad2port(focuslaser_jsonconfig['rfsw00']['conn']['address'], 'D')
            time.sleep(0.1)  # Small delay to ensure switch settles

            # Temporary array for 20 measurements
            temp_measurements = np.zeros(focuslaser_jsonconfig['global_params']['number_averages'])
            for meas_idx in range(focuslaser_jsonconfig['global_params']['number_averages']):
                try:
                    # Single measurement
                    dmm00.write("INIT")
                    time.sleep(0.05)  # Short delay to ensure DMM is ready
                    dmm_data = dmm00.query_ascii_values("FETC?", container=np.array)
                    temp_measurements[meas_idx] = dmm_data[0] if len(dmm_data) > 0 else np.nan
                except Exception as meas_e:
                    print(f"Error in measurement {meas_idx+1} at X={x} um, Z={zpos} um: {meas_e}")
                    temp_measurements[meas_idx] = np.nan

            # Assign temporary measurements to data array
            data['rawdata']['quadD']['dmm00_curr_amp'][idx] = temp_measurements

        except Exception as e:
            print(f"Error measuring photocurrent at X={x} um, Z={zpos} um: {e}")
            data['rawdata']['quadD']['dmm00_curr_amp'][idx] = np.nan
            continue
        '''
    # Save data for this Z position
    fout = '{}_{}_{}_Z{:05d}um'.format(
        data['rawdata']['start_date'],
        data['rawdata']['start_time'],
        save_fname_str,
        int(np.round(zpos, decimals=3) * 1000)
    )

    # Combine dictionaries
    outdict = focuslaser_jsonconfig | data

    # Save the instrument data    
    try:
        with open(pathlib.Path(save_dir_str, fout).with_suffix(fext), 'wb') as fopen:
            pickle.dump(outdict, fopen)
            print("Saving data...\nLocation: {}\nFile Name: {}".format(save_dir_str, fout))
    except Exception as e:
        print(f"Error saving data: {e}")
        with open(pathlib.Path(fout).with_suffix(fext), 'wb') as fopen:
            pickle.dump(outdict, fopen)
            print("Saving data...\nLocation: {}\nFile Name: {}".format(os.getcwd(), fout))
    
    # Plot data for this Z position
    plt.plot(x_array, data['rawdata']['quadD']['dmm00_curr_amp'].mean(axis=1), label='quadD')
    plt.legend()
    plt.title('DC Photocurrent at Z = {:03.5f} mm'.format(zpos))
    plt.xlabel('X Position (um)')
    plt.ylabel('Photocurrent (A)')
    plt.show()

    #zpos += zstep # Increment zposition 
    if zpos < 11:
        zpos -= zstep # Increment zposition 
    elif zpos < 13.5:
        zpos -= 0.3 # Increment zposition 
    else: 
        zpos -= zstep
    
# -------------------- Cleanup -------------------- #
stage.conex.move(lta00, 1, float(focuslaser_jsonconfig['global_params']['zstart_um']), 'abs')
stage.nsc.move(nsc00, 7, focuslaser_jsonconfig['global_params']['xstart_um'])
stage.nsc.move(nsc00, 2, focuslaser_jsonconfig['global_params']['ystart_um'])

print("Experiment complete!")