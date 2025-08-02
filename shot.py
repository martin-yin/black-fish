from ctypes import windll, byref, c_ubyte
from ctypes.wintypes import RECT, HWND
import numpy as np
from win32 import win32api, win32gui, win32print
import ctypes
import psutil
import win32gui
import win32process

GetDC = windll.user32.GetDC
CreateCompatibleDC = windll.gdi32.CreateCompatibleDC
GetClientRect = windll.user32.GetClientRect
CreateCompatibleBitmap = windll.gdi32.CreateCompatibleBitmap
SelectObject = windll.gdi32.SelectObject
BitBlt = windll.gdi32.BitBlt
SRCCOPY = 0x00CC0020
GetBitmapBits = windll.gdi32.GetBitmapBits
DeleteObject = windll.gdi32.DeleteObject
ReleaseDC = windll.user32.ReleaseDC

# 排除缩放干扰
windll.user32.SetProcessDPIAware()

def capture(handle: HWND):
    """窗口客户区截图

    Args:
        handle (HWND): 要截图的窗口句柄

    Returns:
        numpy.ndarray: 截图数据
    """
    # 获取窗口客户区的大小
    r = RECT()
    GetClientRect(handle, byref(r))
    width, height = r.right, r.bottom
    # 开始截图
    dc = GetDC(handle)
    cdc = CreateCompatibleDC(dc)
    bitmap = CreateCompatibleBitmap(dc, width, height)
    SelectObject(cdc, bitmap)
    BitBlt(cdc, 0, 0, width, height, dc, 0, 0, SRCCOPY)
    # 截图是BGRA排列，因此总元素个数需要乘以4
    total_bytes = width*height*4
    buffer = bytearray(total_bytes)
    byte_array = c_ubyte*total_bytes
    GetBitmapBits(bitmap, total_bytes, byte_array.from_buffer(buffer))
    DeleteObject(bitmap)
    DeleteObject(cdc)
    ReleaseDC(handle, dc)
    # 返回截图数据为numpy.ndarray
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)

def find_window_by_process_name(process_name):
    """通过进程名查找窗口"""
    def enum_windows_proc(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                if process.name().lower() == process_name.lower():
                    window_title = win32gui.GetWindowText(hwnd)
                    lParam.append((hwnd, window_title, pid))
            except:
                pass
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_proc, windows)
    return windows

def find_window_by_title_partial(partial_title):
    """通过部分标题查找窗口"""
    def enum_windows_proc(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if partial_title.lower() in window_title.lower():
                lParam.append((hwnd, window_title))
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_proc, windows)
    return windows


def get_game_window_handle():
    """获取游戏窗口句柄"""
    print("开始查找游戏窗口...")
    
    # 方法1: 精确匹配
    print("方法1: 尝试精确匹配窗口标题 'BlackDesert64'")
    hwnd = win32gui.FindWindow(None, "BlackDesert64")
    if hwnd:
        print(f"方法1成功: 找到窗口句柄 {hwnd}")
        return hwnd
    else:
        print("方法1失败: 未找到精确匹配的窗口")
    
    # 方法2: 部分匹配
    print("方法2: 尝试部分匹配窗口标题包含 'BlackDesert'")
    windows = find_window_by_title_partial("BlackDesert")
    if windows:
        print(f"方法2成功: 找到 {len(windows)} 个匹配窗口")
        for i, (hwnd, title) in enumerate(windows):
            print(f"  窗口{i+1}: 句柄={hwnd}, 标题='{title}'")
        return windows[0][0]  # 返回第一个匹配的窗口句柄
    else:
        print("方法2失败: 未找到包含 'BlackDesert' 的窗口")
    
    # 方法3: 通过进程名
    print("方法3: 尝试通过进程名 'BlackDesert64.exe' 查找")
    windows = find_window_by_process_name("BlackDesert64.exe")
    if windows:
        print(f"方法3成功: 找到 {len(windows)} 个匹配进程的窗口")
        for i, (hwnd, title, pid) in enumerate(windows):
            print(f"  窗口{i+1}: 句柄={hwnd}, 标题='{title}', PID={pid}")
        return windows[0][0]
    else:
        print("方法3失败: 未找到 'BlackDesert64.exe' 进程的窗口")
    
    print("所有方法都失败，未找到游戏窗口")
    return None

if __name__ == "__main__":
    import cv2
    hwnd = get_game_window_handle()
    if hwnd:
        print(f"找到窗口句柄: {hwnd}")
        image = capture(hwnd)
        cv2.imwrite("CaptureTest.png", image)
    else:
        print("未找到游戏窗口")