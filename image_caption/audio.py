"""
DEEP LIGHT UPDATED ON 14.01.2019
"""
import vlc
from gtts import gTTS


def play(string):
    tts = gTTS(text=string, lang='en')
    tts.save("out.mp3")
    player = vlc.MediaPlayer("out.mp3")
    player.audio_set_volume(100)
    player.play()



