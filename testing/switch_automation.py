import serial
import pyvisa
import time

# -------------------- Initialise PyVISA -------------------- #
rm = pyvisa.ResourceManager()

# -------------------- DMM config -------------------- #
DMM_ADDRESS = 'TCPIP::192.168.0.6::INSTR'
dmm = rm.open_resource(DMM_ADDRESS)

# Reset the DMM
dmm.write('*RST')

# Send the list of commands to configure the instrument
print('\n-------------------- Configuring Digital Multimeter (DMM) 00 --------------------')
dmm.write('CONF:VOLT:DC')

# Voltage test measurement
voltage = float(dmm.query("READ?"))
print(f"Measured Voltage: {voltage:.2f} V")

# -------------------- Switch Matrix config -------------------- #
# Serial connection for the switch matrix
ser = serial.Serial('COM5', 19200, timeout=1)
#ser.write_termination = ''

# Address the instrument at GPIB 7
ser.write(b'++addr7\r')

# Disable specific status queue events
ser.write(b':stat:que:dis (-410,-410)\n')
time.sleep(0.1)

# Functions to switch connections
def connect_in3_out1(ser):
    ser.write(b':open all\n')
    time.sleep(0.1) # REMOVE
    ser.write(b':clos (@ 1!39)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!40)\n')
    time.sleep(0.1)

def connect_in4_out1(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!38)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!39)\n')
    time.sleep(0.1)

def connect_in3_out2(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!29)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!30)\n')
    time.sleep(0.1)

def connect_in4_out2(ser):
    ser.write(b':open all\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!28)\n')
    time.sleep(0.1)
    ser.write(b':clos (@ 1!29)\n')
    time.sleep(0.1)

# List of connections to automate
connections = [
    ('IN3 to OUT1', connect_in3_out1),
    ('IN4 to OUT1', connect_in4_out1),
    ('IN3 to OUT2', connect_in3_out2),
    ('IN4 to OUT2', connect_in4_out2)
]

# ============================================================ #
#                             Test                             #
# ============================================================ #

# Automate switching and measuring
for name, connect_func in connections:
    print(f"\nSwitching matrix to {name}...")
    connect_func(ser)
    time.sleep(1) 
    
    try:
        # Trigger a measurement and read the voltage 
        dmm.write('INIT')
        time.sleep(0.05)
        voltage = dmm.query_ascii_values('FETC?')[0]
        print(f"Measured voltage at DMM for {name}: {voltage:.4f} V")
    except Exception as e:
        print(f"Error measuring for {name}: {e}")

# Cleanup: Open all switches and close connections
ser.write(b':open all\n')
time.sleep(0.5)
ser.close()
dmm.close()
rm.close()

print("\nScript complete.")