from pygame import mixer
import time
mixer.init()

def command_play(command='D:\Projects\Trinetra\WhatsApp Ptt 2022-11-21 at 10.49.24 AM-[AudioTrimmer.com].mp3'):
    mixer.music.load(command)
    mixer.music.play()
    time.sleep(1)
