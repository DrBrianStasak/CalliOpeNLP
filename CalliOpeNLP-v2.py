# CalliOpeNLP Automatic Voice Collection Tool v2 - Brian Stasak (2026)
# Near real-time voice assessment software with interactive speech collection assistant Calli
# Participant demographic survey; 18 voice tasks; individual spoken instructions per recording; ASR transcripts; degree of compliance; acoustic features
# Includes self-survey Voice Health Index-10
# Utilizes secure off-line Google Whisper 'tiny' ASR model/s

#Please, reference the following in manuscripts and/or presentations:
#Stasak, B., Li, R., Chacon, A., Black, R., & Madill, C., 2026. CalliOpeNLP: A Standalone Digital Health Voice Data Collection Research Tool,
# In: Proc. INTERSPEECH 2026, Sydney, NSW – Australia, pp. 1–5.

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

# Text output (green is 'active' task cue indicator for voice participant)
green = '\033[32m'
italic = '\033[3m'
reset = '\033[0m'

# Automatically gets (month, day, year) time point for audio file labelling; option for more specific time in seconds below
time_point = datetime.datetime.now()
time_point = time_point.strftime("_%m_%d_%Y")
#time_start = time_point.strftime("%H %M %S")

# Metadata of voice participant (user id)
firstname_voice = str(input("Please, enter your FIRST name: "))
lastname_voice = str(input("Please, enter your LAST name: "))
name_voice = str(firstname_voice + "_" + lastname_voice)
age_voice = str(input("What is your current AGE? "))
gender_voice = str(input("What is your GENDER identity (Female, Male, Non-Binary)? "))
height_voice = str(input("What is your HEIGHT in x.xx metres? "))
esl_voice = str(input("Is English your NATIVE first-language (If YES, type 'y'. If NO, type 'n')? "))
accent_voice = str(input("What is your English ACCENT type (American, Australian, British, French, German, Indian, etc)? "))
ai_consent = str(input("Are you okay if your anonymous recordings are used in future AI large-scale voice models (if YES, type 'y'. If NO, type 'n')? "))

# Voice Health Index-10 self-survey questionnaire
print("\nEnter a score from 0 to 4 (0=NEVER; 1=ALMOST NEVER; 2=SOMETIMES; 3=ALMOST ALWAYS; 4=ALWAYS). \nRate the response that indicates how frequently you have the same experience.\n")
vhi_1 = str(input("My voice makes it difficult for people to hear me: "))
vhi_2 = str(input("People have difficulty understanding me in a noisy room: "))
vhi_3 = str(input("My voice difficulties restrict my personal & social life: "))
vhi_4 = str(input("I feel left out of the conversations because of my voice: "))
vhi_5 = str(input("My voice problem causes me to lose income: "))
vhi_6 = str(input("I feel as though I have to strain to produce voice: "))
vhi_7 = str(input("The clarity of my voice is unpredictable: "))
vhi_8 = str(input("My voice problem upsets me: "))
vhi_9 = str(input("My voice makes me feel handicapped: "))
vhi_10 = str(input("People ask, \"What's wrong with your voice?\": "))
vhi_total = (int(vhi_1) + int(vhi_2) + int(vhi_3) + int(vhi_4) + int(vhi_5) + int(vhi_6) + int(vhi_7) + int(vhi_8) + int(vhi_9) + int(vhi_10))

demograph_voice = (f"ParticipantName: {name_voice}", f"ParticipantAge: {age_voice}", f"ParticipantGender: {gender_voice}", f"ParticipantHeight: {height_voice}", f"EnglishNative: {esl_voice}", f"Accent_voice: {accent_voice}", f"AIConsent: {ai_consent}")
vhi_voice = (f"vhi_1: {vhi_1}", f"vhi_2: {vhi_2}", f"vhi_3: {vhi_3}", f"vhi_4: {vhi_4}", f"vhi_5: {vhi_5}", f"vhi_6: {vhi_6}", f"vhi_7: {vhi_7}", f"vhi_8: {vhi_8}", f"vhi_9: {vhi_9}", f"vhi_10: {vhi_10}", f"vhi_total: {vhi_total}")

# Create voice report documentation; stores voice task data (e.g., compliance, ASR transcripts)
voice_report_data = []
voice_report = (name_voice+"_VoiceReport"+time_point+".txt")

voice_report_data.append(demograph_voice)
voice_report_data.append(vhi_voice)

# Task ground truths
rp_gt = "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colours. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow."
vop_gt = "Elliot ate an apple and allowed Andrew another. Each and every avenue is open at eight o'clock. Over on Aston Avenue there is an open air arena. I am in agreement of every aspect of our association. Alan's attitude is overly obnoxious. In April, Adam always attends an extravaganza in Arizona."
cvp_gt = "The blue spot is on the key again. How hard did he hit him. We were away a year ago. We eat eggs every Easter. My mama makes lemon muffins. Peter will keep at the peak. He helped her hurry home. I eat eggs every evening. Papa took a piece of the cake."
mpta_gt = "Ahhhh"
mpts_gt = "Sssss"
mptz_gt = "Zzzzz"
pr_gt = "Ahhhh"
sd_gt = "Ahhhh"
c1to10_gt = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
c80to90_gt = "80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90."
hb_gt = "Happy birthday to you, happy birthday to you, happy birthday dear someone, happy birthday to you!"
ps_gt = "Twinkle twinkle little star, how I wonder what you are, up above the world so high, like a diamond in the sky, twinkle twinkle little star, how I wonder what you are."
mv_gt = "My voice..."
ao_gt = "I think..."

# Recording setup; duration is pre-fixed - depending on task approximate length; confidence factor; ASR model; and pitch range exemplars
rp_duration = 45
vop_duration = 30
cvp_duration = 30
mpt_duration = 35
pr_duration = 10
sd_duration = 7
ct_duration = 15
hb_duration = 20
ps_duration = 25
mv_duration = 45
ao_duration = 45
fs = 44100
model_size = "tiny" # model type: "large-v3" (runs slow); "tiny" (runs quickly); "small" (runs fast); "base"
wordlist = []
wordlist_phrases = []
task_total_ct = []
pr_hl_demo = 'zDemoPRHtoL.wav' # pre-recorded file allows participant to hear good example of pitch range high to low; keep this WAV in code folder directory
pr_lh_demo = 'zDemoPRLtoH.wav' # pre-recorded file allows participant to hear good example of pitch range high to low; keep this WAV in code folder directory

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
    engine.setProperty('rate', 155) # speech rate (words per minute), adjust for faster/slower
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

    # this check compliance for read task only and skips performance task (sung speech not good with ASR)
    if lang_p > confidence_factor and lev_distance > confidence_factor or gt == mpta_gt or gt == mpts_gt or gt == mptz_gt or gt == pr_gt or gt == sd_gt or gt == hb_gt or gt == ps_gt or gt == mv_gt or gt == ao_gt:
        print("\n*** Thank you for completing this task - we will move on to the next voice task ***.\n")
    else:
        print("\n!!! It seems that you did not follow the directions - please, try follow the directions as best you can !!!")


# Voice task 1 - Rainbow Passage
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def rp_task(task1):
    print("\nBeginning Voice Task 1:\nUsing your regular speaking voice please read aloud the short paragraph in green. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically... \n")
    rp_instructions = "\tOkay, your very first task is reading out loud. Using your regular speaking voice please read aloud the short paragraph in green. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    rp_gt = "\t\tWhen the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.\n" \
            "\t\tThe rainbow is a division of white light into many beautiful colours.\n" \
            "\t\tThese take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon.\n" \
            "\t\tThere is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it.\n" \
            "\t\tWhen a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.\n" \

    tts_engine = initialize_engine()
    speak_text(rp_instructions, tts_engine)
    print(f"{green}{italic}{rp_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task1, rp_duration)
    repeat_task(rp_task, task1)


# Voice task 2 - Vowel Onset Phrases
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def vop_task(task2):
    print("\nBeginning Voice Task 2:\nUsing your regular speaking voice please say the sentences in green. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    vop_instructions = "\tYour second task is reading aloud several phrases. Using your regular speaking voice please say the sentences in green. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    vop_gt = "\t\tElliot ate an apple and allowed Andrew another.\n" \
             "\t\t\tEach and every avenue is open at eight o'clock.\n" \
             "\t\t\t\tOver on Aston Avenue there is an open air arena.\n" \
             "\t\t\t\t\tI am in agreement of every aspect of our association.\n" \
             "\t\t\t\t\t\tAlan's attitude is overly obnoxious.\n" \
             "\t\t\t\t\t\t\tIn April, Adam always attends an extravaganza in Arizona.\n" \

    tts_engine = initialize_engine()
    speak_text(vop_instructions, tts_engine)
    print(f"{green}{italic}{vop_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task2, vop_duration)
    repeat_task(vop_task, task2)


# Voice task 3 - Cape V/R Phrases
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def cvp_task(task3):
    print("\nBeginning Voice Task 3:\nWith your regular speaking voice, say the sentences in green. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    cvp_instructions = "\tYour third task is reading aloud some more phrases. With your regular speaking voice, say the sentences in green. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    cvp_gt = "\t\tThe blue spot is on the key again.\n" \
             "\t\t\tHow hard did he hit him.\n" \
             "\t\t\t\tWe were away a year ago.\n" \
             "\t\t\t\t\tWe eat eggs every Easter.\n" \
             "\t\t\t\t\t\tMy mama makes lemon muffins.\n" \
             "\t\t\t\t\t\t\tPeter will keep at the peak.\n" \
             "\t\t\t\t\t\t\t\tHe helped her hurry home.\n" \
             "\t\t\t\t\t\t\t\t\tI eat eggs every evening.\n" \
             "\t\t\t\t\t\t\t\t\t\tPapa took a piece of the cake.\n" \

    tts_engine = initialize_engine()
    speak_text(cvp_instructions, tts_engine)
    print(f"{green}{italic}{cvp_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task3, cvp_duration)
    repeat_task(cvp_task, task3)


# Voice task 4 - Maximum Phonation 1 /a/
# Gives basic instructions; records held vowel
def mpt1_task(task4):
    print("\nBeginning Voice Task 4:\nTake a deep breath and sustain an 'ahh' vowel like in the word 'SAW' for as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    mpt1_instructions = "\tYour fourth task is to record how long you can sustain an 'aaaaahhhhhh' vowel. Take a deep breath and hold an 'aaaaahhhhhh' vowel, like in the word saw, as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n"

    mpta_gt = "\t\tAaaaahhhhhh-------------->\n" \

    tts_engine = initialize_engine()
    speak_text(mpt1_instructions, tts_engine)
    print(f"{green}{italic}{mpta_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task4, mpt_duration)
    repeat_task(mpt1_task, task4)


# Voice task 5 - Maximum Phonation 2 /a/
# Gives basic instructions; records held vowel
def mpt2_task(task5):
    print("\nBeginning Voice Task 5:\nTake a deep breath and sustain an 'ahhh' vowel like in the word 'SAW' for as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    mpt2_instructions = "\tYour fifth task is to again record how long you can sustain an 'aaaaahhhhhh' vowel. Take a deep breath and sustain an 'aaaaahhhhhh' vowel, like in the word saw as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n"

    mpta_gt = "\t\tAaaaahhhhhh-------------->\n" \

    tts_engine = initialize_engine()
    speak_text(mpt2_instructions, tts_engine)
    print(f"{green}{italic}{mpta_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task5, mpt_duration)
    repeat_task(mpt2_task, task5)


# Voice task 6 - Maximum Phonation 3 /a/
# Gives basic instructions; records held vowel
def mpt3_task(task6):
    print("\nBeginning the Voice Task 6:\nTake a deep breath and sustain an 'ahhh' vowel like in the word 'SAW' for as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    mpt3_instructions = "\tYour six task is to record one last time how long you can sustain an 'aaaaahhhhhh' vowel. Take a deep breath and sustain an 'aaaaahhhhhh' vowel, like in the word saw as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n"

    mpta_gt = "\t\tAaaaahhhhhh-------------->\n" \

    tts_engine = initialize_engine()
    speak_text(mpt3_instructions, tts_engine)
    print(f"{green}{italic}{mpta_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task6, mpt_duration)
    repeat_task(mpt3_task, task6)


# Voice task 7 - Maximum Phonation /s/
# Gives basic instructions; records held consonant
def mpts_task(task7):
    print("\nBeginning the Voice Task 7:\nTake a deep breath and hold an 's' consonant as long as you can, like at the end of the word 'SASS'. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    mpts_instructions = "\tYour seventh task is to record how long you can sustain an 's' consonant like in the word sass. Take a deep breath and hold an 'S' as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n"

    mpts_gt = "\t\tSssssss-------------->\n" \

    tts_engine = initialize_engine()
    speak_text(mpts_instructions, tts_engine)
    print(f"{green}{italic}{mpts_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task7, mpt_duration)
    repeat_task(mpts_task, task7)


# Voice task 8 - Maximum Phonation /z/
# Gives basic instructions; records consonant
def mptz_task(task8):
    print("\nBeginning Voice Task 8:\nTime a deep breath and hold an 'z' consonant as long as you can, like at the end of the word 'BUZZ'. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n")
    mptz_instructions = "\tYour eighth task is to record how long you can sustain a 'z' consonant, like at the end of the word buzz. Take a deep breath and hold a 'Z' as long as you can. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically.... \n"

    mptz_gt = "\t\tZzzzzzz-------------->\n" \

    tts_engine = initialize_engine()
    speak_text(mptz_instructions, tts_engine)
    print(f"{green}{italic}{mptz_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task8, mpt_duration)
    repeat_task(mptz_task, task8)


# Voice task 9 - Pitch Range Low to High /a/
# Gives basic instructions; records upwards vowel
def prltoh_task(task9):
    print("\nBeginning Voice Task 9:\nHere is an example of a vowel's pitch glided upwards.\nFrom your normal speaking pitch, glide an 'ahh' vowel UPWARDS to the HIGHEST pitch that you can reach. Going into a falsetto or head voice is fine.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    prltoh_instructions = "\tThat was an example of a vowel's pitch glided upwards. Your ninth task is to raise an 'aaaaahhhhhh' vowel up. From your normal speaking pitch, glide an 'aaaaahhhhhh' vowel upwards to the highest pitch that you can reach. Going into a falsetto or head voice is fine. Press the ENTER key to start your voice recording.\nAfter you are done, please, remain quiet and the recording will stop automatically. \n"

    data, fs = sf.read(pr_lh_demo, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()

    pr_gt = "\t\tAaaaahhhhhh\n" \

    tts_engine = initialize_engine()
    speak_text(prltoh_instructions, tts_engine)
    print(f"{green}{italic}{pr_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task9, pr_duration)
    repeat_task(prltoh_task, task9)


# Voice task 10 - Pitch Range High to Low /a/
# Gives basic instructions; records read aloud speech
def prhtol_task(task10):
    print("\nBeginning Voice Task 10:\nHere is an example of a vowel's pitch glided downwards.\nFrom your normal speaking pitch, glide an 'ahh' vowel DOWNWARDS to the LOWEST pitch that you can reach.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    prltoh_instructions = "\tThat was an example of a vowel's pitch glided downwards. Your tenth task is to lower an 'aaaaahhhhhh' vowel downward. From your normal speaking pitch, glide an 'aaaaahhhhhh' vowel downwards to the lowest pitch that you can reach. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    data, fs = sf.read(pr_hl_demo, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()

    pr_gt = "\t\tAaaaahhhhhh\n" \

    tts_engine = initialize_engine()
    speak_text(prltoh_instructions, tts_engine)
    print(f"{green}{italic}{pr_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task10, pr_duration)
    repeat_task(prhtol_task, task10)


# Voice task 11 - Sound Dynamics Normal /a/
# Gives basic instructions; records read aloud speech
def sdn_task(task11):
    print("\nBeginning Voice Task 11:\nHold a steady pitched 'ahh' vowel at NORMAL speaking volume for 5 seconds. \nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    sdn_instructions = "\tYour eleventh task is to hold a steady pitched 'aaaaahhhhhh' vowel at normal speaking volume for five seconds. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically....  \n"

    sd_gt = "\t\tAaaaahhhhhh\n" \

    tts_engine = initialize_engine()
    speak_text(sdn_instructions, tts_engine)
    print(f"{green}{italic}{sd_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task11, sd_duration)
    repeat_task(sdn_task, task11)

# Voice task 12 - Sound Dynamics Loud /a/
# Gives basic instructions; records read aloud speech
def sdl_task(task12):
    print("\nBeginning Voice Task 12:\nHold the same steady pitched 'ahh' vowel at your LOUDEST speaking volume for 5 seconds.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    sdn_instructions = "\tYour twelfth task is to hold the same steady pitched 'aaaaahhhhhh' vowel at your loudest speaking volume for five seconds. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically....  \n"

    sd_gt = "\t\tAaaaahhhhhh\n" \

    tts_engine = initialize_engine()
    speak_text(sdn_instructions, tts_engine)
    print(f"{green}{italic}{sd_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task12, sd_duration)
    repeat_task(sdl_task, task12)


# Voice task 13 - Sound Dynamics Quiet /a/
# Gives basic instructions; records read aloud speech
def sdq_task(task13):
    print("\nBeginning the Voice Task 13:\nHold the smae steady pitched 'ahh' vowel at the QUIETEST speaking volume you can make without whispering for 5 seconds.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    sdq_instructions = "\tYour thirteenth task is to hold the same steady pitched 'aaaaahhhhhh' vowel at the quietest volume you can make without whispering for five seconds. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically....  \n"

    sd_gt = "\t\tAaaaahhhhhh\n" \

    tts_engine = initialize_engine()
    speak_text(sdq_instructions, tts_engine)
    print(f"{green}{italic}{sd_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task13, sd_duration)
    repeat_task(sdq_task, task13)


# Voice task 14 - Counting 1 to 10
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def c1to10_task(task14):
    print("\nBeginning Voice Task 14:\nCount from 1 to 10 with a small amount of silence in between each number.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    c1to10_instructions = "\tYour fourteenth task is to count from one to ten with a small amount silence in between each number. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically....  \n"

    c1to10_gt = "\t\t1, 2, 3, 4, 5, 6, 7, 8, 9, 10.\n" \

    tts_engine = initialize_engine()
    speak_text(c1to10_instructions, tts_engine)
    print(f"{green}{italic}{c1to10_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task14, ct_duration)
    repeat_task(c1to10_task, task14)


# Voice task 15 - Counting 80 to 90
# Gives basic instructions; records read aloud speech; checks ASR-based transcript against ground truth text (>75% match okay)
def c80to90_task(task15):
    print("\nBeginning Voice Task 15:\nCount from 80 to 90 continuously with no silence gaps.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    c80to90_instructions = "\tYour fifteenth task is to count from eighty to ninety continuously with no silence gaps. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    c80to90_gt = "\t\t80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90.\n" \

    tts_engine = initialize_engine()
    speak_text(c80to90_instructions, tts_engine)
    print(f"{green}{italic}{c80to90_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task15, ct_duration)
    repeat_task(c80to90_task, task15)


# Voice task 16 - Happy Birthday Song
# Gives basic instructions; records read aloud speech
def hb_task(task16):
    print("\nBeginning Voice Task 16:\nSing the song HAPPY BIRTHDAY to anyone you wish.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    hb_instructions = "\tYour sixteenth task is to sing the song Happy Birthday to any person you wish. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    hb_gt = "\t\tHappy birthday to you, happy birthday to you, happy birthday dear <person>, happy birthday to you!\n" \

    tts_engine = initialize_engine()
    speak_text(hb_instructions, tts_engine)
    print(f"{green}{italic}{hb_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task16, hb_duration)
    repeat_task(hb_task, task16)


# Voice task 17 - Popular Song
# Gives basic instructions; records read aloud speech
def ps_task(task17):
    print("\nBeginning Voice Task 17:\nSing any part of a popular song that you know. It can be a radio hit, commercial tune, tv theme song, hymn, or nursery rhyme.\nIf you cannot think of a tune, you can sing Twinkle Twinkle Little Star.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically... \n")
    ps_instructions = "\tYour seventeenth task is to sing part of any popular song that you know. It can be a radio hit, commercial tune, tv theme song, hymn, or nursery rhyme. If you cannot think of a tune, you can sing, twinkle twinkle little star. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    ps_gt = "\t\tTwinkle twinkle little star, how I wonder what you are, up above the world so high, like a diamond in the sky,\n\t\t\ttwinkle twinkle little star, how I wonder what you are.\n" \

    tts_engine = initialize_engine()
    speak_text(ps_instructions, tts_engine)
    print(f"{green}{italic}{ps_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task17, ps_duration)
    repeat_task(ps_task, task17)


# Voice task 18 - My Voice
# Gives basic instructions; records read aloud speech
def mv_task(task18):
    print("\nBeginning Voice Task 18:\nIn your own words, describe your voice health, how you think it feels and sounds. Begin your very first sentence with the two words: MY VOICE.\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n")
    mv_instructions = "\tYour final voice task is to describe your voice. In your own words, describe your voice health, how you think it feels and sounds? Begin your first sentence with the two words, my, voice. Press the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n"

    mv_gt = "\t\tMy voice...\n" \

    tts_engine = initialize_engine()
    speak_text(mv_instructions, tts_engine)
    print(f"{green}{italic}{mv_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task18, mv_duration)
    repeat_task(mv_task, task18)


# Voice task 19 - Automation Opinion
# Gives basic instructions; records read aloud speech
def ao_task(task19):
    print("\nOptional Task:\nIf you wish to provide feedback regarding this automated voice collection process, you can provide your thoughts in an extra recording.\nIn your own words, describe what you think about this automated software...\nDid you think the process was more or less comfortable than having a real person present?\nWere the instructions clear for each task?\nPress the ENTER key to start your voice recording. After you are done, please, remain quiet and the recording will stop automatically.... \n")
    ao_instructions = str(f"\tAlright, {firstname_voice}, you have completed voice task protocol! Your voice contributions will support new voice research, and also help to advance digital AI health applications, like me, Calli. If you wish to provide feedback regarding your automated voice collection process experience, you can provide your valuable thoughts in an extra recording now.\nIf not, just remain quiet.\nPress the ENTER key to start your optional voice recording. The recording will stop automatically.... \n")

    ao_gt = "\t\tI think...\n" \

    tts_engine = initialize_engine()
    speak_text(ao_instructions, tts_engine)
    print(f"{green}{italic}{ao_gt}{reset}")
    press_key = input("Press ENTER key to start your voice recording... ")

    record(task19, ao_duration)
    repeat_task(ao_task, task19)


#################################################################
#################################################################
# Mr. Cali introduction greeting
print("\nContacting Calli, your friendly voice task assistant...")
tts_engine = initialize_engine()
greeting = str(f"Hello {firstname_voice}. I'm Calli, your friendly voice task assistant. Today, there will be 18 total voice task to complete. I'll provide you with instructions per task. Please, wait until I am done talking before you press the ENTER record button. Let's begin your voice audio recordings...")
finish_1 = str(f"Nice work {firstname_voice}! Let's move on to the next thing to do.")
finish_2 = str(f"Alright, we can now move on to the next item.")
finish_3 = str(f"Excellent, let's go to the next item.")
finish_4 = str(f"Good! Let's record some performance based vocal tasks.")
finish_5 = str(f"Nice work.")
finish_6 = str(f"Okay, let's do another.")
finish_7 = str(f"Alright. Let's move on.")
finish_8 = str(f"We are half way done.")
finish_9 = str(f"Let's try another task.")
finish_10 = str(f"The pitch range tasks are completed.")
finish_11 = str(f"Let's try another one that's louder.")
finish_12 = str(f"That was a loud one alright.")
finish_13 = str(f"The three voice dynamic tasks are done.")
finish_14 = str(f"Okay, let's try another counting task.")
finish_15 = str(f"Nice job counting.")
finish_16 = str(f"Everyone knows the birthday song.")
finish_17 = str(f"That's always a nice tune.")
finish_18 = str(f"Thank you for sharing about your voice health.")
finish_19 = str(f"Okay one last optional task.")

speak_text(greeting, tts_engine)

# Run Task 1 - Rainbow Passage; record, label, transcript, and validate compliance
task_1 = "_RP"
rp_task(task_1)
asr_process(task_1, rp_gt)
extract_features(task_1)
print(finish_1)
speak_text(finish_1, tts_engine)

# Run Task 2 - Vowel Onset Phrases; record, label, transcript, and validate compliance
task_2 = "_VOP"
vop_task(task_2)
asr_process(task_2, vop_gt)
extract_features(task_2)
print(finish_2)
speak_text(finish_2, tts_engine)

# Run Task 3 - Cape-V/R Phrases; record, label, transcript, and validate compliance
task_3 = "_CVP"
cvp_task(task_3)
asr_process(task_3, cvp_gt)
extract_features(task_3)
print(finish_3)
speak_text(finish_3, tts_engine)

# Run Task 4 - Maximum Phonation 1 /a/; record, label, and transcript
task_4 = "_MPT1"
mpt1_task(task_4)
asr_process(task_4, mpta_gt)
extract_features(task_4)
print(finish_4)
speak_text(finish_4, tts_engine)

# Run Task 5 - Maximum Phonation 2 /a/; record, label, and transcript
task_5 = "_MPT2"
mpt2_task(task_5)
asr_process(task_5, mpta_gt)
extract_features(task_5)
print(finish_5)
speak_text(finish_5, tts_engine)

# Run Task 6 - Maximum Phonation 3 /a/; record, label, and transcript
task_6 = "_MPT3"
mpt3_task(task_6)
asr_process(task_6, mpta_gt)
extract_features(task_6)
print(finish_6)
speak_text(finish_6, tts_engine)

# Run Task 7 - Maximum Phonation /s/; record, label, and transcript
task_7 = "_MPTS"
mpts_task(task_7)
asr_process(task_7, mpts_gt)
extract_features(task_7)
print(finish_7)
speak_text(finish_7, tts_engine)

# Run Task 8 - Maximum Phonation /z/; record, label, and transcript
task_8 = "_MPTZ"
mptz_task(task_8)
asr_process(task_8, mptz_gt)
extract_features(task_8)
print(finish_8)
speak_text(finish_8, tts_engine)

# Run Task 9 - Pitch Range Low to High; record, label, and transcript
task_9 = "_PRLtoH"
prltoh_task(task_9)
asr_process(task_9, pr_gt)
extract_features(task_9)
print(finish_9)
speak_text(finish_9, tts_engine)

# Run Task 10 - Pitch Range High to Low; record, label, and transcript
task_10 = "_PRHtoL"
prhtol_task(task_10)
asr_process(task_10, pr_gt)
extract_features(task_10)
print(finish_10)
speak_text(finish_10, tts_engine)

# Run Task 11 - Sound Dynamics Normal; record, label, and transcript
task_11 = "_SDN"
sdn_task(task_11)
asr_process(task_11, sd_gt)
extract_features(task_11)
print(finish_11)
speak_text(finish_11, tts_engine)

# Run Task 12 - Sound Dynamics Loud; record, label, and transcript
task_12 = "_SDL"
sdl_task(task_12)
asr_process(task_12, sd_gt)
extract_features(task_12)
print(finish_12)
speak_text(finish_12, tts_engine)

# Run Task 13 - Sound Dynamics Quiet; record, label, and transcript
task_13 = "_SDQ"
sdq_task(task_13)
asr_process(task_13, sd_gt)
extract_features(task_13)
print(finish_13)
speak_text(finish_13, tts_engine)

# Run Task 14 - Counting 1 to 10; record, label, transcript, and validate compliance
task_14 = "_C1to10"
c1to10_task(task_14)
asr_process(task_14, c1to10_gt)
extract_features(task_14)
print(finish_14)
speak_text(finish_14, tts_engine)

# Run Task 15 - Counting 80 to 90; record, label, transcript, and validate compliance
task_15 = "_C80to90"
c80to90_task(task_15)
asr_process(task_15, c80to90_gt)
extract_features(task_15)
print(finish_15)
speak_text(finish_15, tts_engine)

# Run Task 16 - Happy Birthday Song; record, label, and transcript
task_16 = "_HB"
hb_task(task_16)
asr_process(task_16, hb_gt)
extract_features(task_16)
print(finish_16)
speak_text(finish_16, tts_engine)

# Run Task 17 - Popular Song; record, label, and transcript
task_17 = "_PS"
ps_task(task_17)
asr_process(task_17, ps_gt)
extract_features(task_17)
print(finish_17)
speak_text(finish_17, tts_engine)

# Run Task 18 - My Voice; record, label, and transcript
task_18 = "_MV"
mv_task(task_18)
asr_process(task_18, mv_gt)
extract_features(task_18)
print(finish_18)
speak_text(finish_18, tts_engine)

# Run Task 19 - Automation Opinion; how experience was using voice collection software; record, label, and transcript
task_19 = "_AO"
ao_task(task_19)
asr_process(task_19, ao_gt)
extract_features(task_19)
print(finish_19)
speak_text(finish_19, tts_engine)


#################################
#################################
# End of Voice Collection Protocol - write voice report text file
file = open(voice_report, "w")
file.writelines(str(voice_report_data))
file.close()

print("\n*********************** ALL DONE WITH YOUR VOICE DATA COLLECTION ***********************")
print("******************************************************************************************\n")
#print(voice_report_data)




