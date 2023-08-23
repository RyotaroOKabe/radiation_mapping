from matplotlib.colors import to_hex
import numpy as np
def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]

def rgb_to_hex(rgb):
    # Normalize RGB values to be within 0-1 range
    r, g, b = [val / 255.0 for val in rgb]
    # Convert to hexadecimal representation
    hex_color = to_hex((r, g, b))
    return hex_color

def calculate_expectation(xdata, ydata):
    if len(xdata) != len(ydata):
        raise ValueError("Input arrays must have the same length")
    if np.sum(ydata) < 1e-3:
        raise ValueError("Input arrays must have non-zero distributions")
    
    expectation = np.sum(xdata * ydata)/np.sum(ydata)
    
    return expectation