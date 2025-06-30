# AutoEQ-project

This is part of a project for my internship.

The AutoEQ program is a presentable I am making. It will take a song as a .wav file and a speaker's frequency profile as a .csv file

Using python and its libraries of Librosa, numpy and matplotlib, the program will extract a frequency profile of the song.

The program will also read the .csv file taking only the frequencies that are close to the bands defined at the start. It will then use the song's frequency profile as a target, generating an eq curve for the speaker.

--Updates to the README will take place as the project goes on--

30 - 06 - 2025

Created EQ_playback_code.py it gives real time playback with the generated equaliser, right now I cannot pause or choose another song, features I will work on. I also need to link the file with Song_analyser.py so that i can use my generated EQ curve.


