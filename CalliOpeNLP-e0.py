# CalliOpeNLP Automatic Voice Collection Tool e0 - Brian Stasak (2026)
# This is an emotion-paralinguistic data collection version involving surveys and visual stimuli
# Includes 5 voice-speech, pscyho-linguistic, tech-savviness surveys; natural speech picture description monologue recordings (many emotionally-charged)
# The selection of images are mostly famous paintings and Pulitzer photographs; these were selected because they were not too graphic and can elicit a variety of emotions/stories
# Participant demographic surveys; picture description tasks; ASR transcripts; degree of compliance; acoustic features
# Useful in understanding potential links between voice behaviour, psychometics, and paralinguistic / linguistic expression
# Utilizes secure off-line Google Whisper 'tiny' ASR model/s
# A newer upcoming version will include automated affective keyword feature extraction based on transcripts per recording

# Please, reference the following references in manuscripts and/or presentations:
# [1] Stasak, B., Li, R., Chacon, A., Black, R., & Madill, C., 2026. CalliOpeNLP: A Standalone Digital Health Voice Data Collection Research Tool, In: Proc. INTERSPEECH 2026, Sydney, NSW – Australia, pp. 1–5.
# [2] Stasak, B., Speech Elicitation Protocol Extension for Digital Health Screening (SEPEDHS): TXu 2-298-789; -791; -792; -793; -867 ·©Issued: February 7, 2022

# Toolkits required:
import os
import sys
import argparse
import tempfile
import queue
import datetime
import soundfile as sf
import sounddevice as sd
import pandas as pd
import numpy
#assert numpy  # avoid "imported but unused" message (W0611)
from scipy.io.wavfile import write
#import json
import pyttsx3
import fuzzywuzzy.fuzz
import Levenshtein
import fuzzywuzzy
import faster_whisper
from faster_whisper import WhisperModel
import parselmouth
from parselmouth.praat import call
import nltk
import cv2

# Text output (green is 'active' task cue indicator for voice participant)
green = '\033[32m'
italic = '\033[3m'
reset = '\033[0m'

# Automatically gets (month, day, year) time point for audio file labelling; option for more specific time in seconds below
time_point = datetime.datetime.now()
time_point = time_point.strftime("_%m_%d_%Y")
#time_start = time_point.strftime("%H %M %S")

# Introduction
print('Initiating participant demographic information and 5 surveys. This process should take less than 10 minutes to complete.\n')

# Metadata of voice participant (user id)
print("\nDEMOGRAPHIC INFORMATION")
firstname_voice = str(input("Please, enter your FIRST name: "))
lastname_voice = str(input("Please, enter your LAST name: "))
name_voice = str(firstname_voice + "_" + lastname_voice)
age_voice = str(input("What is your current AGE? "))
gender_voice = str(input("What is your GENDER identity (Female, Male, Non-Binary)? "))
height_voice = str(input("What is your HEIGHT in x.xx metres? "))
esl_voice = str(input("Is English your NATIVE first-language (If YES, type 'y'. If NO, type 'n')? "))
accent_voice = str(input("What is your English ACCENT (American, Australian, British, French, Indian, etc)? "))
ai_consent = str(input("Are you okay if your anonymous recordings are used in AI models (if YES, type 'y'. If NO, type 'n')? "))

# Voice Health Index-10 self-survey questionnaire
print("\nVOICE HEALTH QUESTIONNAIRE - 10 Questions")
print("\nEnter a score from 0 to 4 (0=NEVER; 1=ALMOST NEVER; 2=SOMETIMES; 3=ALMOST ALWAYS; 4=ALWAYS). \nRate the response that indicates how frequently you have the same experience.\n")
vhi_1 = str(input("My voice makes it difficult for people to hear me: "))
vhi_2 = str(input("\tPeople have difficulty understanding me in a noisy room: "))
vhi_3 = str(input("My voice difficulties restrict my personal & social life: "))
vhi_4 = str(input("\tI feel left out of the conversations because of my voice: "))
vhi_5 = str(input("My voice problem causes me to lose income: "))
vhi_6 = str(input("\tI feel as though I have to strain to produce voice: "))
vhi_7 = str(input("The clarity of my voice is unpredictable: "))
vhi_8 = str(input("\tMy voice problem upsets me: "))
vhi_9 = str(input("My voice makes me feel handicapped: "))
vhi_10 = str(input("\tPeople ask, \"What's wrong with your voice?\": "))
vhi_total = (int(vhi_1) + int(vhi_2) + int(vhi_3) + int(vhi_4) + int(vhi_5) + int(vhi_6) + int(vhi_7) + int(vhi_8) + int(vhi_9) + int(vhi_10))
print("\nCompleted 1-of-5 Surveys!")

# Self-talk speech formulation and planning questionnaire; psycho-linguistic trait survey based on Stasak, B. (©2022)
print("\n\nSELF-TALK QUESTIONNAIRE - 9 Questions")
print("\nEnter a score from 1 to 5 (1=STRONGLY DISAGREE; 2=DISAGREE; 3=NEUTRAL; 4=AGREE; 5=STRONGLY AGREE). \nReport the value that best represents each question.\n")
stq_1 = str(input("When I think, my thoughts flow like speech requiring little effort: "))
stq_2 = str(input("During conversation, I can construct clear thoughts in my mind and then quickly say them aloud with ease: "))
stq_3 = str(input("When I think internally, I hear only my voice formulating thoughts and no one else’s: "))
stq_4 = str(input("I think with words and fully formed sentences: "))
stq_5 = str(input("When I talk aloud, and then I realize that I made a grammar mistake, I think about how I could correct it: "))
stq_6 = str(input("If my thoughts were typed to create a diary of consciousness, it could be easily read by others like a book: "))
stq_7 = str(input("Often, when I am listening to someone else tell a story, I repeat important parts of his/her story in my mind: "))
stq_8 = str(input("When I think, I used the same vocabulary and expressions that I would when talking to others: "))
stq_9 = str(input("Several times a day, I reflect on events using private thoughts and this process helps me to succeed in life: "))
stq_total = (int(stq_1) + int(stq_2) + int(stq_3) + int(stq_4) + int(stq_5) + int(stq_6) + int(stq_7) + int(stq_8) + int(stq_9))
print("\nCompleted 2-of-5 Surveys!")

# Self-talk memory and emotional processing questionnaire; psycho-linguistic trait survey based on Stasak, B. (©2022)
print("\n\nTHINKING AFFECT QUESTIONNAIRE - 10 Questions")
print("\nEnter a score from 1 to 5 (1=STRONGLY DISAGREE; 2=DISAGREE; 3=NEUTRAL; 4=AGREE; 5=STRONGLY AGREE). \nReport the value that best represents each question.\n")
smq_1 = str(input("I think through emotional things in my mind using well-formulated sentences: "))
smq_2 = str(input("I feel that I have empowered, full control of my inner speech: "))
smq_3 = str(input("I can easily block out negative thoughts and phrase: "))
smq_4 = str(input("In my mind, it is rare for me to hear other people’s voices hassling or bothering me: "))
smq_5 = str(input("When I get flustered by someone or something, I can calm myself down by talking silently to myself: "))
smq_6 = str(input("When I am being competitive, I often think of phrases like, “You can do this!” or “You can win!: "))
smq_7 = str(input("I am usually careful with my words and formulate the basis of what I want to say before I say it: "))
smq_8 = str(input("When I think of a past event, I hear myself retell the story in my mind using detailed sentences: "))
smq_9 = str(input("I use my inner speech to help monitor choices in my physical and emotional behavior: "))
smq_10 = str(input("When alone, I can easily change topics in my mind and talk to myself about other things: "))
smq_total = (int(smq_1) + int(smq_2) + int(smq_3) + int(smq_4) + int(smq_5) + int(smq_6) + int(smq_7) + int(smq_8) + int(smq_9) + int(smq_10))
print("\nCompleted 3-of-5 Surveys!")

# Self-perception; psycho-linguistic trait survey based on Stasak, B. (©2022)
print("\n\nSELF-PERCEPTION QUESTIONNAIRE - 10 Questions")
print("\nEnter a score from 1 to 5 (1=STRONGLY DISAGREE; 2=DISAGREE; 3=NEUTRAL; 4=AGREE; 5=STRONGLY AGREE). \nReport the value that best represents each question.\n")
spq_1 = str(input("I am a person who contributes to society at least of equal value to others around me: "))
spq_2 = str(input("I am a person who has a fair number of talents and virtues: "))
spq_3 = str(input("Although times can get difficult, I am never really inclined to believe that I am a failure: "))
spq_4 = str(input("My general abilities are on par with most other people: "))
spq_5 = str(input("Thus far in life, I feel satisfied about all my accomplishments: "))
spq_6 = str(input("I am a person that maintains a positive attitude about who I am and what I am doing: "))
spq_7 = str(input("I am a person who is generally pleased with myself: "))
spq_8 = str(input("I am a person who respects myself and the decisions I have made: "))
spq_9 = str(input("I am a person that is handy to have around to help others: "))
spq_10 = str(input("Although times can get difficult, I do not think that I am of little value: "))
spq_total = (int(spq_1) + int(spq_2) + int(spq_3) + int(spq_4) + int(spq_5) + int(spq_6) + int(spq_7) + int(spq_8) + int(spq_9) + int(spq_10))
print("\nCompleted 4-of-5 Surveys!")

# Smart device technology questionnaire; survey based on Stasak, B. (©2022)
print("\n\nSMART DEVICE TECHNOLOGY USE QUESTIONNAIRE - 20 Questions")
print("\nEnter a score from 1 to 5 (1=STRONGLY DISAGREE; 2=DISAGREE; 3=NEUTRAL; 4=AGREE; 5=STRONGLY AGREE). \nReport the value that best represents each question.\n")
sdt_1 = str(input("I have good knowledge of how to properly operate smart device technology: "))
sdt_2 = str(input("I have owned a smart device for many years: "))
sdt_3 = str(input("I routinely use a smart device when communicating with others: "))
sdt_4 = str(input("I regularly use a smart device for work: "))
sdt_5 = str(input("I normally use a smart device for gaming, reading, web browsing, and/or shopping: "))
sdt_6 = str(input("I enjoy using smart device apps for everyday things like emails, messaging, banking, and/or web browsing: "))
sdt_7 = str(input("I know how to use a smart device from family, friend, IT staff/sales, and/or online video tutorial: "))
sdt_8 = str(input("When talking on a smart phone or tablet, I prefer to use line-in headset microphone or Bluetooth headphones: "))
sdt_9 = str(input("I prefer to talk on smart device rather than an old-fashioned landline telephone: "))
sdt_10 = str(input("I use or have used a smart device to help monitor my health (e.g., heart rate, distance walked, caloric intake): "))
sdt_11 = str(input("I use or have used speech-to-text on my smart device for text messaging, web searches, and/or obtaining driving directions: "))
sdt_12 = str(input("I have experience using tele-health or online medical consultation (i.e., video meeting with a healthcare professional): "))
sdt_13 = str(input("I have shown a family member or friend how to operate a smart device: "))
sdt_14 = str(input("I have modified my smart device settings due to a visual, hearing, and/or some other impairment or functional reason: "))
sdt_15 = str(input("I am so familiar with a smart phone that I can navigate and text messaging as quick or even quicker than most other people: "))
sdt_16 = str(input("I believe that smart devices can help people monitor their health and provide better access to healthcare professionals: "))
sdt_17 = str(input("I am not nervous about communicating on a smart device due privacy issues and concerns: "))
sdt_18 = str(input("I do not get very frustrated easily using a smart device to communicate: "))
sdt_19 = str(input("I prefer to use sound with video when talking to friends/family on a smart device: "))
sdt_20 = str(input("I have owned a few different smart device manufacturer’s products (e.g., Amazon, Apple, Google, Microsoft, Samsung): "))
sdt_total = (int(sdt_1) + int(sdt_2) + int(sdt_3) + int(sdt_4) + int(sdt_5) + int(sdt_6) + int(sdt_7) + int(sdt_8) + int(sdt_9) + int(sdt_10) + int(sdt_11) + int(sdt_12) + int(sdt_13) + int(sdt_14) + int(sdt_15) + int(sdt_16) + int(sdt_17) + int(sdt_18) + int(sdt_19) + int(sdt_20))
print("\nCompleted 5-of-5 Surveys! Moving on to the Voice Recording Tasks...")


demograph_voice = (f"ParticipantName: {name_voice}", f"ParticipantAge: {age_voice}", f"ParticipantGender: {gender_voice}", f"ParticipantHeight: {height_voice}", f"EnglishNative: {esl_voice}", f"Accent_voice: {accent_voice}", f"AIConsent: {ai_consent}")
vhi_voice = (f"vhi_1: {vhi_1}", f"vhi_2: {vhi_2}", f"vhi_3: {vhi_3}", f"vhi_4: {vhi_4}", f"vhi_5: {vhi_5}", f"vhi_6: {vhi_6}", f"vhi_7: {vhi_7}", f"vhi_8: {vhi_8}", f"vhi_9: {vhi_9}", f"vhi_10: {vhi_10}", f"vhi_total: {vhi_total}")
stq_voice = (f"stq_1: {stq_1}", f"stq_2: {stq_2}", f"stq_3: {stq_3}", f"stq_4: {stq_4}", f"stq_5: {stq_5}", f"stq_6: {stq_6}", f"stq_7: {stq_7}", f"stq_8: {stq_8}", f"stq_9: {stq_9}", f"stq_total: {stq_total}")
smq_voice = (f"smq_1: {smq_1}", f"smq_2: {smq_2}", f"smq_3: {smq_3}", f"smq_4: {smq_4}", f"smq_5: {smq_5}", f"smq_6: {smq_6}", f"smq_7: {smq_7}", f"smq_8: {smq_8}", f"smq_9: {smq_9}", f"smq_10: {smq_10}", f"smq_total: {smq_total}")
spq_voice = (f"spq_1: {spq_1}", f"spq_2: {spq_2}", f"spq_3: {spq_3}", f"spq_4: {spq_4}", f"spq_5: {spq_5}", f"spq_6: {spq_6}", f"spq_7: {spq_7}", f"spq_8: {spq_8}", f"spq_9: {spq_9}", f"spq_10: {spq_10}", f"spq_total: {spq_total}")
sdt_voice = (f"ssdt_1: {sdt_1}", f"sdt_2: {sdt_2}", f"sdt_3: {sdt_3}", f"sdt_4: {sdt_4}", f"sdt_5: {sdt_5}", f"sdt_6: {sdt_6}", f"sdt_7: {sdt_7}", f"sdt_8: {sdt_8}", f"sdt_9: {sdt_9}", f"sdt_10: {sdt_10}", f"sdt_11: {sdt_11}", f"sdt_12: {sdt_12}", f"sdt_13: {sdt_13}", f"sdt_14: {sdt_14}", f"sdt_15: {sdt_15}", f"sdt_16: {sdt_16}", f"sdt_17: {sdt_17}", f"sdt_18: {sdt_18}", f"sdt_19: {sdt_19}", f"sdt_20: {sdt_20}", f"sdt_total: {sdt_total}")

# Create voice report documentation; stores voice task data (e.g., compliance, ASR transcripts)
voice_report_data = []
voice_report = (name_voice+"_VoiceReport"+time_point+".txt")

voice_report_data.append(demograph_voice)
voice_report_data.append(vhi_voice)
voice_report_data.append(stq_voice)
voice_report_data.append(smq_voice)
voice_report_data.append(spq_voice)
voice_report_data.append(sdt_voice)

# Task ground truths - we do not have this except for read sentence tasks where we know exactly what should be said
p1_gt = "I think"
ao_gt = "I think"

# Recording setup; duration is pre-fixed - depending on task approximate length; confidence factor; ASR model; and pitch range exemplars
p_duration = 50
ao_duration = 50
waitkey_time = 45
fs = 44100
model_size = "tiny" # model type: "large-v3" (runs slow); "tiny" (runs quickly); "small" (runs fast); "base"
wordlist = []
wordlist_phrases = []
task_total_ct = []

# Confidence factor setting based on speaker's ESL standing and age range; non-English or older speakers are known to generate higher ASR WER
# This compliance-check setting needs to be systematically tested to determine fair parameter setting; degree of error allowed (dysfluencies, inserts, repeats, deletions)
if esl_voice == "N" or esl_voice == "n" or esl_voice == "No" or esl_voice == "no" or int(age_voice) > 70:
    confidence_factor = 0.60
else:
    confidence_factor = 0.75

# Calli's pre-set accent setting - this is test demo only... stdies show that listener's prefer similar accent to their own
# Voice Types: Indian-English Male (4); UK-English Male/Female (23/38); French-English Male (28); Slavic-English Female (36/85); UAE-English Female (52)
# US-English Male Old (70); Italian-English Female (2); Clean Robot Male (51); Singing Robot Voice Male (17); Voice Disordered Male/female (14/21)
if gender_voice == 'Female' or gender_voice == 'female' or gender_voice == 'f':
    if accent_voice == 'British' or accent_voice == 'Australian' or accent_voice == 'Irish' or accent_voice == 'New Zealand':
        calli_accent = 38
    if accent_voice == 'French':
        calli_accent = 28
    if accent_voice == 'Italian' or accent_voice == 'Spanish' or accent_voice == 'Brazilian' or accent_voice == 'Portuguese':
        calli_accent = 2
    if accent_voice == 'Russian' or accent_voice == 'Polish' or accent_voice == 'Ukrainian':
        calli_accent = 36
    if accent_voice == 'Indonesian' or accent_voice == 'Iranian' or accent_voice == 'Egyptian' or accent_voice == 'Saudi':
        calli_accent = 52
    if accent_voice == 'Indian' or accent_voice == 'Pakistani' or accent_voice == 'Sri Lankan':
        calli_accent = 4
else:
    calli_accent = 4

if gender_voice == 'Male' or gender_voice == 'male' or gender_voice == 'm':
    if accent_voice == 'American' or accent_voice == 'Canadian':
        calli_accent = 70
    if accent_voice == 'British' or accent_voice == 'Australian' or accent_voice == 'Irish' or accent_voice == 'New Zealand':
        calli_accent = 23
    if accent_voice == 'French':
        calli_accent = 28
    if accent_voice == 'Indian' or accent_voice == 'Pakistani' or accent_voice == 'Sri Lankan':
        calli_accent = 4
else:
    calli_accent = 4

if gender_voice == 'Non-Binary' or gender_voice == 'non-binary' or gender_voice == 'nb':
    calli_accent = 4


# This process allows a participant to redo any voice task by entering any form of yes
def repeat_task(task, taskn):
    decision_voice = str(input("Would you like to redo the last task? If YES, type 'y'. If NO, type 'n'. Then press ENTER. "))
    if decision_voice == 'Y' or decision_voice == 'Yes' or decision_voice == 'yes' or decision_voice == 'y':
        task_ct = 1 # one indicates task attempt repeated
        task_total_ct.append(task_ct)
        task(taskn) # This redoes the task that it was on
    else:
        task_ct = 0 # zero indicates task attempt correctly completed
        task_total_ct.append(task_ct)
        return print('Proceeding to the next voice task.')


# Recording process - dtype must be 'float32'
def record(task, duration):
    print("\n\t\t<<<!!!!!! RECORDING NOW !!!!!!>>>\n")
    voice_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    write(name_voice+time_point+task+".wav", fs, voice_data)
    press_key = input("Press the ENTER key when ready to move on to the next voice task... ")
    return print("Finished your voice recording! \n")


# Text to speech setup
def initialize_engine():
    engine = pyttsx3.init() # initializes the text-to-speech engine
    engine.setProperty('rate', 160) # speech rate (words per minute), adjust for faster/slower
    engine.setProperty('volume', 0.95) # speech volume (0.0 to 1.0)
    voices = engine.getProperty('voices') # get list of available voices
    if voices:
        engine.setProperty('voice', voices[calli_accent].id) # setup voice-type [4, 14, 17, 21] are okay; or use calli_accent variable
    else:
        print("Warning: No voices found for pyttsx3. Using default system voice.")
    return engine


# Text to speech process
def speak_text(text, engine): # reads in text-to-speech engine parameters
    engine.say(text)      # queue the predetermined text to be spoken
    engine.runAndWait()   # pauses code until spoken content is finished
    del engine # this clears engine so it is ready for next task


# ASR system; run on CPU locally; beam_size = larger takes longer but wider keyword search (3 to 5 okay)
def asr_process(task, gt):
    model = WhisperModel(model_size, device="cpu")
    segments, info = model.transcribe((name_voice+time_point+task+".wav"), beam_size=3)
    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    lang_prob = info.language_probability
    voice_report_data.append(task)

    for segment in segments:
        wordlist.append(segment.text)
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        wordlist_phrase = str("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        wordlist_phrases.append(wordlist_phrase)

    transcript = "".join(wordlist)
    transcript_times = "".join(wordlist_phrases)
    #print(transcript)
    #print(transcript_times)
    compliance(lang_prob, gt, transcript, transcript_times, task_total_ct)
    wordlist.clear() # must clear temporary transcript after each task for proper compliance scoring
    wordlist_phrases.clear()  # must clear temporary transcript after each task for proper compliance scoring
    return


# Praat acoustic feature extraction process
def extract_features(task):
    audio_file = (name_voice+time_point+task+".wav")
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch()
    intensity = sound.get_intensity()
    #harmonicity = sound.to_harmonicity()
    acoustic_features = (str(pitch),"AverageIntensity_dB: ", str(intensity))
    voice_report_data.append(acoustic_features)
    return


# Compliance determines if participant said what he/she was supposed to (utilizes relative approximate overlap threshold)
def compliance(prob, gt, task_transcript, transcript_times, task_total_ct):
    lang_p = prob
    #print(f'\nEnglish Language Probability: {lang_p:.2f}')
    lev_distance = Levenshtein.ratio(gt, task_transcript)
    #print(f'Levenshtein Distance: {lev_distance:.2f}')
    token_set_ratio = fuzzywuzzy.fuzz.token_set_ratio(gt, task_transcript)
    #print(f'Token Set Ratio: {token_set_ratio}')
    voice_data = (f"EnglishProbability: {lang_p}", f"LevenshteinDistance: {lev_distance}", f"TokenSetRatio: {token_set_ratio}", f"ASRTranscript: {task_transcript}", f"ASRTranscriptPhraseTimes: {transcript_times}", f"TaskTotalCount: {task_total_ct}")
    voice_report_data.append(voice_data)

    # this check compliance is only used for read task and skips performance/free speech tasks
    if lang_p > confidence_factor:
        print("\n*** Thank you for completing this task - we will move on to the next voice task ***.\n")
    else:
        print("\n!!! It seems that you did not follow the directions - please, try follow the directions as best you can !!!")


# Voice task 1 - Picture 1 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p1_task(task1):
    print("\nBeginning Voice Task 1:\nDescribe the following painting in your own words. What do you think is happening? What emotion do you think is being conveyed. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p1_instructions = "\tDescribe the following painting in your own words. What do you think is happening? What emotion do you think is being conveyed? Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p1_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p1_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/1_OctaveTassaert-SleepingChild-1850.JPEG'
    img_1 = cv2.imread(p1_image)
    cv2.imshow('Picture 1 of 14', img_1)
    cv2.waitKey(waitkey_time)

    record(task1, p_duration)
    repeat_task(p1_task, task1)


# Voice task 2 - Picture 2 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p2_task(task2):
    print("\nBeginning Voice Task 2:\nDescribe the next painting in your own words. Provide any details on what you think is happening and what emotion is being conveyed. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p2_instructions = "\tDescribe the next painting in your own words. Provide any details on what you think is happening and what emotion is being conveyed. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p2_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p2_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/2_QuentinMatsys-MoneyLenderWife-1890.JPEG'
    img_2 = cv2.imread(p2_image)
    cv2.imshow('Picture 2 of 14', img_2)
    cv2.waitKey(waitkey_time)

    record(task2, p_duration)
    repeat_task(p2_task, task2)


# Voice task 3 - Picture 3 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p3_task(task3):
    print("\nBeginning Voice Task 3:\nDescribe the next painting again in your own words. Provide any details on what you think is being conveyed. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p3_instructions = "\tDescribe the next painting in your own words and any important details that you think is being conveyed. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p3_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p3_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/3_BertholdWoltze-IrritatingGentleman-1874.JPEG'
    img_3 = cv2.imread(p3_image)
    cv2.imshow('Picture 3 of 14', img_3)
    cv2.waitKey(waitkey_time)

    record(task3, p_duration)
    repeat_task(p3_task, task3)

# Voice task 4 - Picture 4 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p4_task(task4):
    print("\nBeginning Voice Task 4:\nAgain, describe the next painting in your own words and how it makes you feel. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p4_instructions = "\tAgain, describe the next painting in your own words and how it makes you feel. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p4_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p4_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/4_BertholdWoltze-TheLetter-1896.JPEG'
    img_4 = cv2.imread(p4_image)
    cv2.imshow('Picture 4 of 14', img_4)
    cv2.waitKey(waitkey_time)

    record(task4, p_duration)
    repeat_task(p4_task, task4)

# Voice task 5 - Picture 5 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p5_task(task5):
    print("\nBeginning Voice Task 5:\nExplain what you think is happening in the next painting. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p5_instructions = "\tExplain what you think is happening in the next painting. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p5_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p5_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/5_GaetanoPreviati-PaoloFrancesca-1887.JPEG'
    img_5 = cv2.imread(p5_image)
    cv2.imshow('Picture 5 of 14', img_5)
    cv2.waitKey(waitkey_time)

    record(task5, p_duration)
    repeat_task(p5_task, task5)

# Voice task 6 - Picture 6 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p6_task(task6):
    print("\nBeginning Voice Task 6:\nExplain what you think the story is in the next painting. Provide any details on what you think is happening. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p6_instructions = "\tExplain what you think the story is in the next painting. Provide any details on what you think is happening. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p6_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p6_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/6_LukeFildes-TheDoctor-1891.JPEG'
    img_6 = cv2.imread(p6_image)
    cv2.imshow('Picture 6 of 14', img_6)
    cv2.waitKey(waitkey_time)

    record(task6, p_duration)
    repeat_task(p6_task, task6)

# Voice task 7 - Picture 7 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p7_task(task7):
    print("\nBeginning Voice Task 7:\nWhat do you think has happened in this painting and how would you explain its mood? The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p7_instructions = "\tWhat do you think has happened in this painting and how would you explain its mood? Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p7_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p7_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/7_NikolayKasatkin-Who-1897.JPEG'
    img_7 = cv2.imread(p7_image)
    cv2.imshow('Picture 7 of 14', img_7)
    cv2.waitKey(waitkey_time)

    record(task7, p_duration)
    repeat_task(p7_task, task7)

# Voice task 8 - Picture 8 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p8_task(task8):
    print("\nBeginning Voice Task 8:\nDescribe the following photograph in your own words. Provide any details and what emotion you think is being conveyed. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p8_instructions = "\tDescribe the following photograph in your own words. Provide any details and what emotion you think is being conveyed. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p8_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p8_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/8_EdwardCurtis-IndianSummer-1890.JPEG'
    img_8 = cv2.imread(p8_image)
    cv2.imshow('Picture 8 of 14', img_8)
    cv2.waitKey(waitkey_time)

    record(task8, p_duration)
    repeat_task(p8_task, task8)

# Voice task 9 - Picture 9 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p9_task(task9):
    print("\nBeginning Voice Task 9:\nDescribe the next photograph. What emotion do you think is being conveyed and why? The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p9_instructions = "\tDescribe the next photograph. What emotion do you think is being conveyed and why? Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p9_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p9_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/9_JohnLGaunt-TragedybytheSea-1955.JPEG'
    img_9 = cv2.imread(p9_image)
    cv2.imshow('Picture 9 of 14', img_9)
    cv2.waitKey(waitkey_time)

    record(task9, p_duration)
    repeat_task(p9_task, task9)

# Voice task 10 - Picture 10 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p10_task(task10):
    print("\nBeginning Voice Task 10:\nDescribe the next photograph. What do you think it is trying to say? The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p10_instructions = "\tDescribe the next photograph. What do you think it is trying to say? Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p10_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p10_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/10_WilliamCBeall-Faith&Confidence-1957.JPEG'
    img_10 = cv2.imread(p10_image)
    cv2.imshow('Picture 10 of 14', img_10)
    cv2.waitKey(waitkey_time)

    record(task10, p_duration)
    repeat_task(p10_task, task10)

# Voice task 11 - Picture 11 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p11_task(task11):
    print("\nBeginning Voice Task 11:\nDescribe what is happening in the following photograph. Provide any details on what you think is happening and what emotion you think is being conveyed. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p11_instructions = "\tDescribe what is happening in the following photograph. Provide any details on what you think is happening and what emotion you think is being conveyed. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p11_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p11_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/11_DonMcCullin-TheBogside-1971.JPEG'
    img_11 = cv2.imread(p11_image)
    cv2.imshow('Picture 11 of 14', img_11)
    cv2.waitKey(waitkey_time)

    record(task11, p_duration)
    repeat_task(p11_task, task11)

# Voice task 12 - Picture 12 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p12_task(task12):
    print("\nBeginning Voice Task 12:\nDescribe this next photograph as best as you can. The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p12_instructions = "\tDescribe this next photograph as best as you can. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p12_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p12_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/12_StanleyJForman-BostonCityFall-1975.JPEG'
    img_12 = cv2.imread(p12_image)
    cv2.imshow('Picture 12 of 14', img_12)
    cv2.waitKey(waitkey_time)

    record(task12, p_duration)
    repeat_task(p12_task, task12)

# Voice task 13 - Picture 13 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p13_task(task13):
    print("\nBeginning Voice Task 13:\nDescribe the next photograph in your own words. Provide any details about the photo and where you think it might be. The recording will stop automatically after minute. Press the ENTER key to start your voice recording.\n")
    p13_instructions = "\tDescribe this photograph in your own words and any details about the photo and where you think it might be. Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p13_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p13_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/13_BrianStasak-Winery-2016.JPEG'
    img_13 = cv2.imread(p13_image)
    cv2.imshow('Picture 13 of 14', img_13)
    cv2.waitKey(waitkey_time)

    record(task13, p_duration)
    repeat_task(p13_task, task13)

# Voice task 14 - Picture 14 Description
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def p14_task(task14):
    print("\nBeginning Voice Task 14:\nDescribe the final photograph in your own words. Where do you think this was taken? The recording will stop automatically after a minute. Press the ENTER key to start your voice recording.\n")
    p14_instructions = "\tDescribe the final photograph in your own words. Where do you think this was taken? Press the ENTER key to start your voice recording. \n"

    tts_engine = initialize_engine()
    speak_text(p14_instructions, tts_engine)

    press_key = input("Press ENTER key to start your voice recording... ")

    p14_image = '<YOURDIRCTORYGOESHERE>/AffectImagesResized/14_BrianStasak-OpticalRecursion-2009.JPEG'
    img_14 = cv2.imread(p14_image)
    cv2.imshow('Picture 14 of 14', img_14)
    cv2.waitKey(waitkey_time)

    record(task14, p_duration)
    repeat_task(p14_task, task14)


# Voice task 15 - Automation Opinion
# Gives basic instructions; records read aloud speech
def ao_task(task15):
    print("\nOptional Task:\nIf you wish to provide feedback regarding this automated voice collection process, you can provide your thoughts in an extra recording.\nIn your own words, describe what you think about this automated voice collection software...\nDid you think the process was more or less comfortable than having a real person present?\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n")
    ao_instructions = str(f"\tAlright, {firstname_voice}, you have completed all the voice recording tasks! Your contributions will support new voice research, and also help to understand spoken language. If you wish to provide feedback regarding your automated voice collection process experience, you can provide your valuable thoughts in an extra recording now.\nIf not, just remain quiet.\nPress the ENTER key to start your optional voice recording. The recording will stop automatically.... \n")

    ao_gt = "\t\tI think...\n" \

    tts_engine = initialize_engine()
    speak_text(ao_instructions, tts_engine)
    print(f"{green}{italic}{ao_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task15, ao_duration)
    repeat_task(ao_task, task15)

#################################################################
#################################################################
# Mr. Cali introduction greeting
print("\nContacting Calli, your friendly voice task assistant...")
tts_engine = initialize_engine()
greeting = str(f"Hello {firstname_voice}. I'm Calli, your friendly voice recording task assistant. Today, there will be 14 picture description voice tasks to complete. Please, wait until I am done talking before you press the ENTER record button. And, remember, all recordings will stop automatically after about a minute... Let's begin your voice recordings...")
finish = str(f"Nice work {firstname_voice}! Let's move on to the next one.")

speak_text(greeting, tts_engine)

# Run Task 1 - Rainbow Passage; record, label, transcript, and validate compliance
task_1 = "_P1"
p1_task(task_1)
asr_process(task_1, p1_gt)
extract_features(task_1)
print(finish)
speak_text(finish, tts_engine)

# Run Task 2 - Rainbow Passage; record, label, transcript, and validate compliance
task_2 = "_P2"
p2_task(task_2)
asr_process(task_2, p1_gt)
extract_features(task_2)
print(finish)
speak_text(finish, tts_engine)

# Run Task 3 - Rainbow Passage; record, label, transcript, and validate compliance
task_3 = "_P3"
p3_task(task_3)
asr_process(task_3, p1_gt)
extract_features(task_3)
print(finish)
speak_text(finish, tts_engine)

# Run Task 4 - Rainbow Passage; record, label, transcript, and validate compliance
task_4 = "_P4"
p4_task(task_4)
asr_process(task_4, p1_gt)
extract_features(task_4)
print(finish)
speak_text(finish, tts_engine)

# Run Task 5 - Rainbow Passage; record, label, transcript, and validate compliance
task_5 = "_P5"
p5_task(task_5)
asr_process(task_5, p1_gt)
extract_features(task_5)
print(finish)
speak_text(finish, tts_engine)

# Run Task 6 - Rainbow Passage; record, label, transcript, and validate compliance
task_6 = "_P6"
p6_task(task_6)
asr_process(task_6, p1_gt)
extract_features(task_6)
print(finish)
speak_text(finish, tts_engine)

# Run Task 7 - Rainbow Passage; record, label, transcript, and validate compliance
task_7 = "_P7"
p7_task(task_7)
asr_process(task_7, p1_gt)
extract_features(task_7)
print(finish)
speak_text(finish, tts_engine)

# Run Task 8 - Rainbow Passage; record, label, transcript, and validate compliance
task_8 = "_P8"
p8_task(task_8)
asr_process(task_8, p1_gt)
extract_features(task_8)
print(finish)
speak_text(finish, tts_engine)

# Run Task 9 - Rainbow Passage; record, label, transcript, and validate compliance
task_9 = "_P9"
p9_task(task_9)
asr_process(task_9, p1_gt)
extract_features(task_9)
print(finish)
speak_text(finish, tts_engine)

# Run Task 10 - Rainbow Passage; record, label, transcript, and validate compliance
task_10 = "_P10"
p10_task(task_10)
asr_process(task_10, p1_gt)
extract_features(task_10)
print(finish)
speak_text(finish, tts_engine)

# Run Task 11 - Rainbow Passage; record, label, transcript, and validate compliance
task_11 = "_P11"
p11_task(task_11)
asr_process(task_11, p1_gt)
extract_features(task_11)
print(finish)
speak_text(finish, tts_engine)

# Run Task 12 - Rainbow Passage; record, label, transcript, and validate compliance
task_12 = "_P12"
p12_task(task_12)
asr_process(task_12, p1_gt)
extract_features(task_12)
print(finish)
speak_text(finish, tts_engine)

# Run Task 13 - Rainbow Passage; record, label, transcript, and validate compliance
task_13 = "_P13"
p13_task(task_13)
asr_process(task_13, p1_gt)
extract_features(task_13)
print(finish)
speak_text(finish, tts_engine)

# Run Task 14 - Rainbow Passage; record, label, transcript, and validate compliance
task_14 = "_P14"
p14_task(task_14)
asr_process(task_14, p1_gt)
extract_features(task_14)
print(finish)
speak_text(finish, tts_engine)

# Run Task 15 - Automation Opinion; how experience was using voice collection software; record, label, and transcript
task_15 = "_AO"
ao_task(task_15)
asr_process(task_15, ao_gt)
extract_features(task_15)
print(finish)

#################################
#################################
# End of Voice Collection Protocol - write voice report text file
file = open(voice_report, "w")
file.writelines(str(voice_report_data))
file.close()

print("\n*********************** ALL DONE WITH YOUR VOICE DATA COLLECTION ***********************")
print("******************************************************************************************\n")
#print(voice_report_data)




