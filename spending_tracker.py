from tkinter import Tk, Label
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# from utils import loadConfig
import threading
from pynput import keyboard

from utils.spend import fancy_round


class SpendingWindow:
    def __init__(self, money_dir):
        self.window = Tk()
        self.window.geometry("+%d+0" % (self.window.winfo_screenwidth() - 200))
        self.window.attributes('-topmost', 1)  # Makes the window always on top
        self.money_dir = money_dir
        self.labels = []

        event_handler = FileSystemEventHandler()
        event_handler.on_modified = self.on_modified
        self.observer = Observer()
        self.observer.schedule(event_handler, self.money_dir, recursive=False)
        self.observer.start()

    def toggle_visibility(self):
        if self.window.winfo_viewable():
            self.window.withdraw()
        else:
            self.window.deiconify()

    def readSpending(self):
        spending_by_day = {}
        with open( self.money_dir ) as f:
            for line in f:
                date, time, model, prompt_tokens, prompt_cost, continuation_tokens, continuation_cost = line.split()
                if date not in spending_by_day:
                    spending_by_day[date] = {}
                if model not in spending_by_day[date]:
                    spending_by_day[date][model] = 0

                spending_by_day[date][model] += float(prompt_cost) + float(continuation_cost)
                
        today = datetime.now().strftime("%Y-%m-%d")
        report = "\n".join([f"{model}: ${fancy_round(spending_by_day[today][model])}" for model in spending_by_day.get(today, [])])
        data = f"""
Spending for {today}:
{report}
"""
        return data

    def updateSpending(self):
        spending_data = self.readSpending()
        for label in self.labels:
            label.destroy()
        self.labels = []
        label = Label(self.window, text=spending_data)
        label.pack()
        self.labels.append(label)


    def on_modified(self, event):
        if event.src_path == self.money_dir:
            self.window.after(100, self.updateSpending) # Schedule update on main thread

    def run(self):
        self.updateSpending()
        self.window.mainloop()
        self.observer.stop()
        self.observer.join()

# cfg = loadConfig()
import sys
if len(sys.argv) > 1:
    money_dir = sys.argv[1]
else:
    money_dir = "./logs/spending_records.txt"
spendingWindow = SpendingWindow(money_dir)


ctrl_pressed = False
shift_pressed = False
# await_reset = False

def _on_press(key):
    global ctrl_pressed, shift_pressed#, await_reset
    # Listen for Ctrl + `
    if key == keyboard.Key.ctrl:
        ctrl_pressed = True
    elif key == keyboard.Key.shift:
        shift_pressed = True
    elif key == keyboard.KeyCode(char=':') and ctrl_pressed and shift_pressed:# and not await_reset:
        spendingWindow.toggle_visibility()
        # await_reset = True

def _on_release(key):
    global ctrl_pressed, shift_pressed#, await_reset
    # Listen for Ctrl + `
    if key == keyboard.Key.ctrl:
        ctrl_pressed = False
    elif key == keyboard.Key.shift:
        shift_pressed = False
        
    # if await_reset and not ctrl_pressed and not shift_pressed:
    #     await_reset = False

def listen():
    with keyboard.Listener(
        on_press=_on_press,
        on_release=_on_release) as listener:
        listener.join()

t = threading.Thread(target=listen)
t.start()

spendingWindow.run()