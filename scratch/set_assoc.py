import sys
import winreg

py_exe = sys.executable
cmd_str = f'"{py_exe}" "%1" %*'

try:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.py") as k:
        winreg.SetValue(k, "", winreg.REG_SZ, "Python.File")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Python.File\shell\open\command") as k:
        winreg.SetValue(k, "", winreg.REG_SZ, cmd_str)

    print(f"Successfully associated .py with {py_exe}")
except Exception as e:
    print(f"Error: {e}")
