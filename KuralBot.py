import os
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from operator import itemgetter
from collections import Counter
import random
from tkinter import ttk 
from tkinter import *
import regex
import json
from kural import Thirukkural
thirukkural = __import__('thirukkural')
cdeeplearn = __import__("cdeeplearn")

"""  Message Patterns """
_RE_DICT = {
    'contains': regex.compile(r"^\s?contains\s?(?P<contains>[\p{L}*+|\p{L}\p{M}*+].*)\s?.*", regex.IGNORECASE),
    'ends_with': regex.compile(r"^\s?ends\s?with\s?(?P<ends_with>[\p{L}*+|\p{L}\p{M}*+].*)\s?.*", regex.IGNORECASE),
    'ends_with_1': regex.compile(r"^\s?(?P<ends_with>[\p{L}*+|\p{L}\p{M}*+])\s?[என]?\s?முடியும்.*", regex.IGNORECASE),
    'starts_with': regex.compile(r"^\s?starts\s?with\s?(?P<starts_with>[\p{L}*+|\p{L}\p{M}*+].*)\s?.*", regex.IGNORECASE),
    'starts_with_1': regex.compile(r"^\s?(?P<starts_with>[\p{L}*+|\p{L}\p{M}*+])\s?[எனத்]?\s?தொடங்கும்.*", regex.IGNORECASE),
    'get':regex.compile(r"^\s?get\s?(?P<Chapter>\d+)?\s?,?\s?(?P<Verse>\d+)?\s?.*", regex.IGNORECASE),
    'Kural':regex.compile(r"^\s?Kural|குறள்\s?(?P<Kural>\d+)?.*", regex.IGNORECASE),
    'Help':regex.compile(r"^\s?Help|உதவி\s?.*", regex.IGNORECASE),
    'Quit':regex.compile(r"^\s?Quit|End|Thanks|Bye|நன்றி\s?.*", regex.IGNORECASE),
    'Greet':regex.compile(r"^\s?Welcome|Greet|Hello|வணக்கம்|வாழ்த்து|நல்வரவு\s?.*", regex.IGNORECASE),
    'New':regex.compile(r"^\s?New|Generate|Create|புதிய|வாழ்த்து|உருவாக்கு\s?.*", regex.IGNORECASE),
    }
class KuralBot():
    config = {}
    def __init__(self,bot_config_file = "./KuralBot.json"):
        self.mainApp = Tk()
        self.BotText = None
        self.EntryBox = None
        self.tk = thirukkural.Thirukural()
        with open(bot_config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.initialize_KuralBot()
    def get_bot_parameters(self, bot_config_file=None):
        with open(bot_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    def initialize_KuralBot(self):
        self.mainApp.title(self.config['CHATBOT_TITLE'])
        self.mainApp.geometry(self.config["geometry"])#"450x600")
        self.mainApp.resizable(width=FALSE, height=FALSE)
        self.BotText = Text(self.mainApp, bd=0, bg=self.config["bot_text_bg_color"], height=self.config["bot_text_height"], 
                             width=self.config["bot_text_width"], font=self.config["bot_text_font"],)
        self.BotText.config(state=DISABLED)
        #Bind scrollbar to Chat window
        scrollbar = Scrollbar(self.mainApp, command=self.BotText.yview, cursor=self.config["bot_cursor"])
        self.BotText['yscrollcommand'] = scrollbar.set        
        #Create Button to send message
        SendButton = Button(self.mainApp, font=self.config["button_text_font"], text=self.config['BUTTON_TEXT'], 
                            width=self.config["button_text_width"], height=self.config["button_text_height"],
                            bd=0, bg=self.config["button_text_bg_color"], activebackground=self.config["button_text_activebackground"],
                            fg=self.config["button_text_fg_color"], command= self.send )
        #Create the box to enter message
        self.EntryBox = Text(self.mainApp, bd=0, bg=self.config["entrybox_text_bg_color"],width=self.config["entrybox_text_width"], 
                             height=self.config["entrybox_text_height"], font=self.config["entrybox_text_font"])
        self.EntryBox.bind(sequence="<Return>", func=self.send, add=None)
        #Place all components on the screen
        self.EntryBox.place(x=self.config["entrybox_x"], y=self.config["entrybox_y"], height=self.config["entrybox_height"], width=self.config["entrybox_width"])
        scrollbar.place(x=self.config["scrollbar_x"], y=self.config["scrollbar_y"], height=self.config["scrollbar_height"])
        self.BotText.place(x=self.config["bot_app_x"], y=self.config["bot_app_y"], height=self.config["bot_app_height"], width=self.config["bot_app_width"])
        SendButton.place(x=self.config["button_x"], y=self.config["button_y"], height=self.config["button_height"])
        self._update_with_bot_response("Welcome") 
        self._update_with_bot_response("Help") 
        self.EntryBox.focus_set()
    def show(self):       
        self.mainApp.mainloop()
    def quit(self):
        self.mainApp.after(self.config["wait_time_msec_before_closing_chat_window"],self.mainApp.quit)
    def _update_with_user_message(self,msg):
        self.BotText.tag_config("user_msg", foreground=self.config['USER_MSG_COLOR'])
        self.BotText.tag_config("user_name", foreground=self.config['USER_NAME_COLOR'])
        if msg != '':
            self.BotText.config(state=NORMAL)
            self.BotText.insert(END, self.config['USER_NAME'],'user_name')
            self.BotText.insert(END, msg + '\n\n','user_msg')
            self.BotText.config(foreground="#446665", font=("Verdana", 12 ))
            self.BotText.config(state=DISABLED)
            self.BotText.yview(END)
    def _update_with_bot_response(self, msg):
        self.BotText.tag_config("bot_msg", foreground=self.config['BOT_MSG_COLOR'])
        self.BotText.tag_config("bot_name", foreground=self.config['BOT_NAME_COLOR'])
        res = ''
        if msg != '':
            self.BotText.config(state=NORMAL)
            res = self.get_response(msg)
            self.BotText.insert(END, self.config['BOT_NAME'],'bot_name')
            self.BotText.insert(END, res + '\n\n','bot_msg')
            self.BotText.config(state=DISABLED)
            self.BotText.yview(END)
        return res
    def send(self,event=None):
        msg = self.EntryBox.get("1.0",'end-1c').strip()
        self.EntryBox.delete("0.0",END)
        self._update_with_user_message(msg)
        res = self._update_with_bot_response(msg)
        #"""
        if res == self.config["QUIT_MSG"]:
            self.quit()
        #"""
    def get_response(self, user_message):
        ### implement get respomse
        key, match = self._parse_line(user_message)
        response = ""
        if key == "contains":
            word = match.group("contains")
            response = self.tk.contains(word)
        elif key == "ends_with" or key == "ends_with_1":
            word = match.group("ends_with")
            response = self.tk.endswith(word)
        elif key == "starts_with" or key == "starts_with_1":
            word = match.group("starts_with")
            response = self.tk.startswith(word)
        elif key == "get":
            if match.group("Chapter")==None:
                response = self.tk.get()
                return response
            else:
                chapter_number = int(match.group("Chapter"))
            verse_number = None
            if match.group("Verse")==None:
                response = self.tk.get(chapter_number)
            else:
                verse_number = int(match.group("Verse"))
                response = self.tk.get(chapter_number, verse_number)
        elif key == "Kural":
            if match.group("Kural") is None:
                kural_number = None
            else:
                kural_number = int(match.group("Kural"))
            response = self.tk.get(kural_number=kural_number)     
        elif key == "New":
            response = cdeeplearn.generate_tokens_from_corpus(corpus_files=['thirukural1.txt'], 
                    length=7, save_to_file='kural_model.h5',perform_training=False)
        elif key == "Help":
            response = self.config["HELP_MSG"]
        elif key == "Greet":
            response = self.config["GREET_MSG"]
        elif key == "Quit":
            response = self.config["QUIT_MSG"]
        else:
            response = self.config['FALLBACK_MSG']
        return response
    def _parse_line(self, line):
        for key, rx in _RE_DICT.items():
            match = rx.search(line)
            if match:
                return key, match
        # if there are no matches
        return None, None

if __name__ == "__main__":
    """ Main script """
    #"""
    kb = KuralBot()
    kb.show()
    exit()
    #"""  
