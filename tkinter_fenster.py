# Import necessary libraries
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont


class TkPrinter:
    """Simple non-blocking logger window for showing print output in a Tkinter ScrolledText."""
    def __init__(self, title="Log", font=("Consolas", 19)):
        self.root = tk.Tk()
        self.root.title(title)

        # Accept either a (family, size) tuple or an integer size
        if isinstance(font, (tuple, list)) and len(font) >= 2:
            family, size = font[0], font[1]
        else:
            try:
                family, size = "Consolas", int(font)
            except Exception:
                family, size = "Consolas", 19

        # normal and bold fonts
        self.tk_font = tkfont.Font(family=family, size=size)
        self.tk_font_bold = tkfont.Font(family=family, size=size, weight='bold')

        self.text = ScrolledText(self.root, state='disabled', width=150, height=80, font=self.tk_font)
        # configure a tag that uses the bold font
        self.text.tag_configure('bold', font=self.tk_font_bold)
        self.text.pack(fill='both', expand=True)

    def write(self, msg: str):
        self.text.configure(state='normal')
        # remember insertion start to limit search to newly inserted text (fallback to start)
        try:
            insert_start = self.text.index(f"end - {len(msg)}c")
        except Exception:
            insert_start = '1.0'

        self.text.insert('end', msg)

        # Make lines that contain "Task <number>" bold (whole line)
        try:
            start = insert_start
            while True:
                pos = self.text.search(r'^.*Task\s*\d+.*$', start, stopindex='end', regexp=True)
                if not pos:
                    break
                end_pos = f"{pos} lineend"
                self.text.tag_add('bold', pos, end_pos)
                start = self.text.index(f"{end_pos} + 1c")
        except Exception:
            # falls etwas mit Suche/Tagging schiefgeht, weiter machen ohne Absturz
            pass

        self.text.see('end')
        self.text.configure(state='disabled')
        # keep UI responsive without blocking main thread
        self.root.update_idletasks()

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass