# -*- coding: utf-8 -*-
# CalliOpeNLP Voice Report ACOUSTIC FEATURES GETTER - Brian Stasak 2026 v0
# This batch folder code quickly extracts and prints selected acoustic features (w/header info)
# It is meant to be useful to extract/organize multiple participant CalliOpen Voice Reports chosen info

# Required Python toolkit imports
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Where your folder with all the participants' CalliOpeNLP Voice Reports are located
directory = "/<DIRECTORYLINKWHEREYOURVOICEREPORTSARE>/"

# Main headers; the feature can be any specific one that you are interested in getting
print("\nFile_NAMEID , AGE , Gender , HEIGHT_KGS , VOICE_TASK, ENGLISH_PROBABILITY , LEVENSHTEIN_DISTANCE, FEATURE")

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            for line in f:
                # This is where you note the exact item that you are looking for listed in the directory's file order)
                # You can change the task (_RP, _AO, ); and switch out value 7 for other feature types 'ERB\nAverage', 'AverageIntensity_dB'
                values = re.findall(r"ParticipantName:\s*'?(.*?)(?:'|,|\))", line)
                for value1 in values:
                    v1 = value1
                    values2 = re.findall(r"ParticipantAge:\s*'?(.*?)(?:'|,|\))", line)
                    for value2 in values2:
                        v2 = value2
                    values3 = re.findall(r"ParticipantGender:\s*'?(.*?)(?:'|,|\))", line)
                    for value3 in values3:
                        v3 = value3
                    values4 = re.findall(r"ParticipantHeight:\s*'?(.*?)(?:'|,|\))", line)
                    for value4 in values4:
                        v4 = value4
                    values5 = re.findall(r"\'_RP\'", line) #name specific voice task '_S\*\'
                    for value5 in values5:
                        v5 = value5
                    values6 = re.findall(r"EnglishProbability:\s*'?(.*?)(?:'|,|\))", line)
                    for value6 in values6:
                        v6 = value6
                    values7 = re.findall(r"LevenshteinDistance:\s*'?(.*?)(?:'|,|\))", line)
                    for value7 in values7:
                        v7 = value7
                    values8 = re.findall(r"ERB\\nAverage:\s*(.*?)\sHz", line)
                    for value8 in values8:
                        v8 = value8
        print(v1,',', v2,',',v3,',',v4,',',v5,',',v6,',',v7,',',v8)
