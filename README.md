# AutoEQ-project

This is part of a project for my internship.

The AutoEQ program is a presentable I am making. It will take a song as a .wav file and a speaker's frequency profile as a .csv file

Using python and its libraries of Librosa, numpy and matplotlib, the program will extract a frequency profile of the song.

The program will also read the .csv file taking only the frequencies that are close to the bands defined at the start. It will then use the song's frequency profile as a target, generating an eq curve for the speaker.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Github Repo where i pulled headphone data from: https://github.com/MS141103/Profile-Builder-Contour/blob/main/README.md
Github Repo which I used to help write the GUI: https://github.com/RamadanIbrahem98/sound-equalizer/tree/main#about-the-project
Github Repo I took inspiration from: https://github.com/mayank12gt/Audio-Equalizer

--Updates to the README will take place as the project goes on--

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SongAnalyser.py is the core of the project. It takes a .WAV file and a csv file for the headphones. Right now SongAnalyser.py has been set to use 10 bands.
A FFT is used to break down the song into its constituent frequencies, next we select the 10 band frequencies from the csv file. Then we comput the band energy for those bands and compare the profile to a target response. EQ settings are generated to change things within 1 octave to the target.

PyQTGUI.py is a dialog box that lets you choose a .WAV file and csv file from your laptop, then you can apply the autoeq and play the song with manual slider control and a graph to display the eq.

The remaining files were drafts to build upto to PyQTGUI, which is a heavily modified version from RamadanIbrahem98
