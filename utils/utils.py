from matplotlib.colors import to_hex
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