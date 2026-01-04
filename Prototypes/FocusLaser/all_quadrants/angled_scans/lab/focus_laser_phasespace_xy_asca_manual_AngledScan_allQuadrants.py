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
json_config_path = '../config/final_config_files_(allQuadrants)'
json_config_fname = 'focus_laser_config_phasespace_xy_asca_manual_DualDiagonals.json'
json_config_fullpath = pathlib.Path((root_dir / json_config_path).resolve(), json_config_fname)

# Data save root path
data_save_root = pathlib.Path('C:/Users/virgorun/Projects/LISA/daq/experiments/Students/ascanio/data/Run1/AllQuadrants')

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
    delta_time = datetime.datetime.strptime(end, strfmt) - datetime.datetime.strptime(start, strfmt)
    delta_totalseconds = delta_time.total_seconds()
    delta_year, remain_year = divmod(delta_totalseconds, 31536000)
    delta_days, remain_days = divmod(remain_year, 86400)
    delta_hours, remain_hours = divmod(remain_days, 3600)
    delta_minutes, remain_minutes = divmod(remain_hours, 60)
    
    print('Duration: {:02d} Years, {:02d} Days, {:02d} Hours, {:02d} Minutes, {:02d} Seconds\n'.format(
        int(delta_year), int(delta_days), int(delta_hours), int(delta_minutes), int(remain_minutes)))
    
    return int(delta_year), int(delta_days), int(delta_hours), int(delta_minutes), int(remain_minutes)

def calcDuration(lenx, leny, navg=1, *args):
    '''
    Estimate the duration of the script
    '''
    est_time_sec = lenx * leny * 10
    for i in args:
        est_time_sec += (lenx * leny * i * navg)
    delta_year, remain_year = divmod(est_time_sec, 31536000)
    delta_days, remain_days = divmod(remain_year, 86400)
    delta_hours, remain_hours = divmod(remain_days, 3600)
    delta_minutes, remain_minutes = divmod(remain_hours, 60)
    
    print('Duration: {:02d} Years, {:02d} Days, {:02d} Hours, {:02d} Minutes, {:02d} Seconds\n'.format(
        int(delta_year), int(delta_days), int(delta_hours), int(delta_minutes), int(remain_minutes)))
    
    return int(delta_year), int(delta_days), int(delta_hours), int(delta_minutes), int(remain_minutes)

def erf_fit(x, a, b, c, d):
    return a * np.erf(b * (x - c)) + d

def erfc_fit(x, a, b, c, d):
    return 1 - erf_fit(x, a, b, c, d)

def beam_waist(x):
    return (1 / (x * np.sqrt(2)))

def dbm2w(x, rbw):
    return 1.0 * (10.0 ** (x / 10.0)) / 1E3

def mfunc(x, a, b, c):
    '''
    Define the M Squared function
    '''
    return a * np.sqrt(1 + ((x - c) ** 2 * 1056E-9 ** 2 * b ** 4) / (np.pi ** 2 * a ** 4))

def mfunc2(x, a, c):
    '''
    Define the M Squared function with fixed M^2
    '''
    return a * np.sqrt(1 + ((x - c) ** 2 * 1056E-9 ** 2 * 1.0 ** 4) / (np.pi ** 2 * a ** 4))

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
    dmm00 = rm.open_resource(focuslaser_jsonconfig['dmm00']['conn']['address'], access_mode=0, send_end=True)
    dmm00.write("*RST")
    print('\n-------------------- Configuring Digital Multimeter (DMM) 00 --------------------')
    pvu.pyvisa_cmd_list(dmm00, focuslaser_jsonconfig['dmm00']['cmd'], 'w')
    dmm00.write("CONF:CURR:DC")
    voltage = float(dmm00.query("READ?"))
    print(f"Measured Voltage: {voltage:.2f} V")
    current = float(dmm00.query("READ?"))
    print(f"Measured Photocurrent: {current:.7f} A")

# -------------------- Switch Matrix config -------------------- #
ser = serial.Serial('COM5', 19200, timeout=1)
ser.write(b'++addr7\r')
ser.write(b':stat:que:dis (-410,-410)\n')
time.sleep(0.1)

def connect_quadA(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!1)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!5)\n')
    time.sleep(0.1)

def connect_quadB(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!2)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!5)\n')
    time.sleep(0.1)

def connect_quadC(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!3)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!5)\n')
    time.sleep(0.1)

def connect_quadD(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!4)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!5)\n')
    time.sleep(0.1)

# -------------------- Y Stage config -------------------- #
print("Trying to open LTA00 stage on:", focuslaser_jsonconfig['lta00']['conn']['address'])
if 'lta00' in focuslaser_jsonconfig.keys():
    print('\n-------------------- Configuring Y Stage (LTA) 00 --------------------')
    lta00 = stage.conex.connect(rm,
                                focuslaser_jsonconfig['lta00']['conn']['address'],
                                focuslaser_jsonconfig['lta00']['conn']['baud_rate'],
                                focuslaser_jsonconfig['lta00']['conn']['channel'],
                                focuslaser_jsonconfig['lta00']['conn']['query_delay'],
                                focuslaser_jsonconfig['lta00']['conn']['force'])
    lta00.write('1OR')
    ypos = stage.conex.move(lta00, 1, float(focuslaser_jsonconfig['global_params']['diagonals'][0]['along_start_mm']), 'abs')

# -------------------- X Stage config -------------------- #
print("Trying to open TRAC00 stage on:", focuslaser_jsonconfig['lta01']['conn']['address'])
if 'lta01' in focuslaser_jsonconfig.keys():
    print('\n-------------------- Configuring X Stage (TRAC) 00 --------------------')
    lta01 = stage.conex.connect(rm,
                                focuslaser_jsonconfig['lta01']['conn']['address'],
                                focuslaser_jsonconfig['lta01']['conn']['baud_rate'],
                                focuslaser_jsonconfig['lta01']['conn']['channel'],
                                focuslaser_jsonconfig['lta01']['conn']['query_delay'],
                                focuslaser_jsonconfig['lta01']['conn']['force'])
    lta01.write('1OR')
    xpos = stage.conex.move(lta01, 1, float(focuslaser_jsonconfig['global_params']['diagonals'][0]['intercept_mm'] +
                                           focuslaser_jsonconfig['global_params']['diagonals'][0]['slope'] *
                                           focuslaser_jsonconfig['global_params']['diagonals'][0]['along_start_mm']), 'abs')

# -------------------- Configure scan parameters -------------------- #

# Define the two diagonals from previous analysis
diagonals = [
    {'name': 'Diagonal_1_AD_BC', 'slope': 0.93745, 'intercept_mm': 1.96105},  # Average of A-D (1.8711) and B-C (2.0510)
    {'name': 'Diagonal_2_AB_DC', 'slope': -0.9604, 'intercept_mm': 6.6808}   # Average of A-B (6.6178) and D-C (6.7438)
]

# Loop over each diagonal
for diag in focuslaser_jsonconfig['global_params']['diagonals']:
    slope = diag['slope']
    intercept_mm = diag['intercept_mm']
    diag_name = diag['name']
    
    # Calculate perpendicular slope and unit vector
    perp_slope = -1 / slope
    norm = np.sqrt(1 + perp_slope ** 2)
    unit_perp_x = 1 / norm
    unit_perp_y = perp_slope / norm
    
    # Scan parameters from config
    along_start = diag['along_start_mm']
    along_stop = diag['along_stop_mm']
    along_step_size = diag['along_step_mm']
    along_steps = int((along_stop - along_start) / along_step_size) + 1
    along_array = np.linspace(along_start, along_stop, along_steps)
    
    u_start = diag['u_start_mm']
    u_stop = diag['u_stop_mm']
    u_step_size = diag['u_step_mm']
    u_steps = int((u_stop - u_start) / u_step_size) + 1
    u_array = np.linspace(u_start, u_stop, u_steps)
    
    # -------------------- Save location config -------------------- #
    save_dir_base = pathlib.Path('{}/{}/DualDiagonalScan/{}_{}_{}_{}_{}_{}_251002_LB1471C_quadABCD_manual_setup_Z50mm_{}'.format(
        data_save_root,
        focuslaser_jsonconfig['dut']['batch'],
        focuslaser_jsonconfig['dut']['foundry'],
        focuslaser_jsonconfig['dut']['wafer'],
        focuslaser_jsonconfig['dut']['dut_type'],
        focuslaser_jsonconfig['dut']['size'],
        focuslaser_jsonconfig['dut']['gap'],
        focuslaser_jsonconfig['dut']['dut_id'],
        diag_name
    ))
    
    if not save_dir_base.exists():
        save_dir_base.mkdir(parents=True, exist_ok=True)
    
    save_fname_str = "{}_{}_{}_{}_{}_{}_{}_QPDGapScan_KnifeEdge_251002_LB1471C_quadABCD_manual_setup_Z50mm_{}".format(
        focuslaser_jsonconfig['dut']['batch'],
        focuslaser_jsonconfig['dut']['foundry'],
        focuslaser_jsonconfig['dut']['wafer'],
        focuslaser_jsonconfig['dut']['dut_type'],
        focuslaser_jsonconfig['dut']['gap'],
        focuslaser_jsonconfig['dut']['gap'],
        focuslaser_jsonconfig['dut']['dut_id'],
        diag_name
    )
    
    print(f"Base save directory for {diag_name}: {save_dir_base.as_posix()}")
    print(f"Save filename for {diag_name}: {save_fname_str}")
    
    # -------------------- Main experiment -------------------- #
    
    # Insert Z position (in micrometers)
    z_position_um = 5000  
    
    for along in along_array:
        x_center = intercept_mm + slope * along
        y_center = along
        
        save_dir_str = save_dir_base / f"Along{along*1000:04.0f}um"
        if not save_dir_str.exists():
            save_dir_str.mkdir(parents=True, exist_ok=True)
        
        data = {
            'rawdata': {
                'comp_gps_time': np.zeros_like(u_array),
                'stage_laser_xposition': np.zeros_like(u_array),
                'stage_laser_yposition': np.zeros_like(u_array),
                'u_position': u_array,
                'quadA': {'dmm00_curr_amp': np.zeros((len(u_array), focuslaser_jsonconfig['global_params']['number_averages']))},
                'quadB': {'dmm00_curr_amp': np.zeros((len(u_array), focuslaser_jsonconfig['global_params']['number_averages']))},
                'quadC': {'dmm00_curr_amp': np.zeros((len(u_array), focuslaser_jsonconfig['global_params']['number_averages']))},
                'quadD': {'dmm00_curr_amp': np.zeros((len(u_array), focuslaser_jsonconfig['global_params']['number_averages']))}
            },
            'global_params': {
                'xstart_mm': x_center + u_array[0] * unit_perp_x,
                'xstop_mm': x_center + u_array[-1] * unit_perp_x,
                'xstep_mm': u_step_size * unit_perp_x,
                'ystart_mm': y_center + u_array[0] * unit_perp_y,
                'ystop_mm': y_center + u_array[-1] * unit_perp_y,
                'ystep_mm': u_step_size * unit_perp_y,
                'slope': slope,
                'intercept_mm': intercept_mm
            }
        }
        
        data['rawdata']['start_date'] = datetoday('%Y%m%d')
        data['rawdata']['start_time'] = timenow('%H%M%S')
        
        for idx, u in enumerate(u_array):
            x_target = x_center + u * unit_perp_x
            y_target = y_center + u * unit_perp_y
            
            try:
                xpos = stage.conex.move(lta01, 1, float(x_target), 'abs')
            except visa.VisaIOError as e:
                print(f"Error moving X stage to {x_target} mm: {e}")
                lta01.clear()
                time.sleep(1)
                continue
            
            try:
                ypos = stage.conex.move(lta00, 1, float(y_target), 'abs')
            except visa.VisaIOError as e:
                print(f"Error moving Y stage to {y_target} mm: {e}")
                lta00.clear()
                time.sleep(1)
                continue
            
            data['rawdata']['stage_laser_xposition'][idx] = xpos
            data['rawdata']['stage_laser_yposition'][idx] = ypos
            
            navg = focuslaser_jsonconfig['global_params']['number_averages']
            
            # Measure Quadrant A
            try:
                print(f"\nSwitching matrix to Quadrant A...")
                connect_quadA(ser)
                time.sleep(1)
                temp_measurements_a = np.zeros(navg)
                for i in range(navg):
                    try:
                        dmm00.write("INIT")
                        time.sleep(0.05)
                        temp_measurements_a[i] = dmm00.query_ascii_values("FETC?")[0]
                    except Exception as e:
                        print(f"Error in single measurement {i+1} at u={u} mm, along={along} mm for Quadrant A: {e}")
                        temp_measurements_a[i] = np.nan
                data['rawdata']['quadA']['dmm00_curr_amp'][idx] = temp_measurements_a
                print(f"along={along} mm, u={u} mm, DC photocurrent Quadrant A = {np.mean(temp_measurements_a):.7f} A")
            except Exception as e:
                print(f"Error measuring at along={along} mm, u={u} mm for Quadrant A: {e}")
                data['rawdata']['quadA']['dmm00_curr_amp'][idx] = np.nan
            
            # Measure Quadrant B
            try:
                print(f"\nSwitching matrix to Quadrant B...")
                connect_quadB(ser)
                time.sleep(1)
                temp_measurements_b = np.zeros(navg)
                for i in range(navg):
                    try:
                        dmm00.write("INIT")
                        time.sleep(0.05)
                        temp_measurements_b[i] = dmm00.query_ascii_values("FETC?")[0]
                    except Exception as e:
                        print(f"Error in single measurement {i+1} at u={u} mm, along={along} mm for Quadrant B: {e}")
                        temp_measurements_b[i] = np.nan
                data['rawdata']['quadB']['dmm00_curr_amp'][idx] = temp_measurements_b
                print(f"along={along} mm, u={u} mm, DC photocurrent Quadrant B = {np.mean(temp_measurements_b):.7f} A")
            except Exception as e:
                print(f"Error measuring at along={along} mm, u={u} mm for Quadrant B: {e}")
                data['rawdata']['quadB']['dmm00_curr_amp'][idx] = np.nan
            
            # Measure Quadrant C
            try:
                print(f"\nSwitching matrix to Quadrant C...")
                connect_quadC(ser)
                time.sleep(1)
                temp_measurements_c = np.zeros(navg)
                for i in range(navg):
                    try:
                        dmm00.write("INIT")
                        time.sleep(0.05)
                        temp_measurements_c[i] = dmm00.query_ascii_values("FETC?")[0]
                    except Exception as e:
                        print(f"Error in single measurement {i+1} at u={u} mm, along={along} mm for Quadrant C: {e}")
                        temp_measurements_c[i] = np.nan
                data['rawdata']['quadC']['dmm00_curr_amp'][idx] = temp_measurements_c
                print(f"along={along} mm, u={u} mm, DC photocurrent Quadrant C = {np.mean(temp_measurements_c):.7f} A")
            except Exception as e:
                print(f"Error measuring at along={along} mm, u={u} mm for Quadrant C: {e}")
                data['rawdata']['quadC']['dmm00_curr_amp'][idx] = np.nan
            
            # Measure Quadrant D
            try:
                print(f"\nSwitching matrix to Quadrant D...")
                connect_quadD(ser)
                time.sleep(1)
                temp_measurements_d = np.zeros(navg)
                for i in range(navg):
                    try:
                        dmm00.write("INIT")
                        time.sleep(0.05)
                        temp_measurements_d[i] = dmm00.query_ascii_values("FETC?")[0]
                    except Exception as e:
                        print(f"Error in single measurement {i+1} at u={u} mm, along={along} mm for Quadrant D: {e}")
                        temp_measurements_d[i] = np.nan
                data['rawdata']['quadD']['dmm00_curr_amp'][idx] = temp_measurements_d
                print(f"along={along} mm, u={u} mm, DC photocurrent Quadrant D = {np.mean(temp_measurements_d):.7f} A")
            except Exception as e:
                print(f"Error measuring at along={along} mm, u={u} mm for Quadrant D: {e}")
                data['rawdata']['quadD']['dmm00_curr_amp'][idx] = np.nan
        
        # Save data
        fout = '{}_{}_{}_Along{:04d}um_Z{:05d}um'.format(
            data['rawdata']['start_date'],
            data['rawdata']['start_time'],
            save_fname_str,
            int(np.round(along, decimals=3) * 1000),
            int(np.round(z_position_um, decimals=3))
        )
        
        outdict = focuslaser_jsonconfig | data
        
        try:
            with open(pathlib.Path(save_dir_str, fout).with_suffix('.pkl'), 'wb') as fopen:
                pickle.dump(outdict, fopen)
                print(f"Saving data...\nLocation: {save_dir_str}\nFile Name: {fout}")
        except Exception as e:
            print(f"Error saving data: {e}")
            with open(pathlib.Path(fout).with_suffix('.pkl'), 'wb') as fopen:
                pickle.dump(outdict, fopen)
                print(f"Saving data...\nLocation: {os.getcwd()}\nFile Name: {fout}")
        
        # Plot data
        plt.plot(u_array, data['rawdata']['quadA']['dmm00_curr_amp'].mean(axis=1), label='quadA')
        plt.plot(u_array, data['rawdata']['quadB']['dmm00_curr_amp'].mean(axis=1), label='quadB')
        plt.plot(u_array, data['rawdata']['quadC']['dmm00_curr_amp'].mean(axis=1), label='quadC')
        plt.plot(u_array, data['rawdata']['quadD']['dmm00_curr_amp'].mean(axis=1), label='quadD')
        plt.legend()
        plt.title(f'DC Photocurrent at along={along} mm ({diag_name})')
        plt.xlabel('u Position along perpendicular (mm)')
        plt.ylabel('Photocurrent (A)')
        plt.show()

# -------------------- Cleanup -------------------- #
stage.conex.move(lta01, 1, float(focuslaser_jsonconfig['global_params']['diagonals'][0]['intercept_mm'] +
                                 focuslaser_jsonconfig['global_params']['diagonals'][0]['slope'] *
                                 focuslaser_jsonconfig['global_params']['diagonals'][0]['along_start_mm']), 'abs')
stage.conex.move(lta00, 1, float(focuslaser_jsonconfig['global_params']['diagonals'][0]['along_start_mm']), 'abs')
ser.write(b':open all\n')
time.sleep(0.5)
ser.close()
rm.close()

print("Experiment complete!")