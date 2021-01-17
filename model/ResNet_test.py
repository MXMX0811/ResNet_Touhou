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
import ctypes
import torch
import torchvision.transforms as transforms
import ResNet

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def get_window_pos(name):
    name = name
    win_handle = win32gui.FindWindow(0, name)
    # 获取窗口句柄
    if win_handle == 0:
        return None
    else:
        return win32gui.GetWindowRect(win_handle), win_handle


# 使用键盘扫描码控制上下左右
def move(act):
    key = [0x48, 0x50, 0x4D, 0x4B]  # up/down/left/right
    press = [key[i] for i, e in enumerate(act) if e == 1]
    if len(press) != 0:
        for i in press:
            PressKey(i)
        time.sleep(0.02)
        for i in press:
            ReleaseKey(i)
    else:
        time.sleep(0.02)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = ResNet.ResNet18()
resnet.load_state_dict(torch.load('ResNet_1.pt'))
resnet.to(device)
resnet.eval()
(x1, y1, x2, y2), handle = get_window_pos('搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil')
# win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0) # 还原最小化窗口
text = win32gui.SetForegroundWindow(handle)  # 使窗口显示在最前面

time.sleep(2)

PressKey(0x1C)  # 按回车进入游戏
time.sleep(0.1)
ReleaseKey(0x1C)

time.sleep(1)

PressKey(0x1C)
time.sleep(0.1)
ReleaseKey(0x1C)

time.sleep(1)

PressKey(0x1C)  # 再次按回车选择难度
time.sleep(0.1)
ReleaseKey(0x1C)

time.sleep(1)

PressKey(0x1C)  # 选择人物
time.sleep(0.1)
ReleaseKey(0x1C)

time.sleep(1)

PressKey(0x1C)  # 选择灵符
time.sleep(0.1)
ReleaseKey(0x1C)

# 使用键盘扫描码控制上下左右
while True:
    img = pyautogui.screenshot(region=[x1 + 40, y1 + 50, x2 - x1 - 320, y2 - y1 - 70])  # x,y,w,h
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (122, 141))
    transf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )
    img = transf(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        img = img.to(device)
        action = resnet(img)
        action *= 100
        action = torch.squeeze(action)
    action = action.cpu().numpy().tolist()
    print(action)
    action = [1 if i > 0.5 else 0 for i in action]
    # 按z子弹
    PressKey(0x2C)
    time.sleep(0.01)
    ReleaseKey(0x2C)
    move(action)

