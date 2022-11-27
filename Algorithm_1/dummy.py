import pygame
from time import sleep
pygame.mixer.init()
#setup music
track = "D:\Projects\Trinetra\WhatsApp Ptt 2022-11-21 at 10.49.24 AM-[AudioTrimmer.com].mp3"
pygame.mixer.music.load(track)
status=0
while True:
	if status==0:
		print("Playing Music")
		pygame.mixer.music.play()
	if pygame.mixer.music.get_busy():
		status=1
		print('busy')
	
	else:
		status=0
		print("One loop completed")