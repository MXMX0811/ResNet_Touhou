# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import win32gui
import pyautogui
import cv2
import numpy as np
import time
import csv
import keyboard

key_list = [0, 0, 0, 0]  # 上、下、左、右


def get_window_pos(name):
    name = name
    win_handle = win32gui.FindWindow(0, name)
    # 获取窗口句柄
    if win_handle == 0:
        return None
    else:
        return win32gui.GetWindowRect(win_handle), win_handle


def print_pressed_keys(e):
    global key_list
    key_list = [0, 0, 0, 0]
    # 上、下、左、右
    get_key_list = [code for code in keyboard._pressed_events]
    if 72 in get_key_list:
        key_list[0] = 1
    if 80 in get_key_list:
        key_list[1] = 1
    if 75 in get_key_list:
        key_list[2] = 1
    if 77 in get_key_list:
        key_list[3] = 1


(x1, y1, x2, y2), handle = get_window_pos('搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil')
# win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0) # 还原最小化窗口
text = win32gui.SetForegroundWindow(handle)  # 使窗口显示在最前面

time.sleep(15)

i = 0
done = False

while not done:
    img = pyautogui.screenshot(region=[x1 + 40, y1 + 50, x2 - x1 - 320, y2 - y1 - 70])  # x,y,w,h
    img = cv2.resize(img, (122, 141))
    img.save('../Dataset/Capture_10/' + str(i) + '.jpg')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    keyboard.hook(print_pressed_keys)
    print(key_list)
    with open("../Dataset/KeyCapture_10.csv", "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(key_list)
    i += 1


