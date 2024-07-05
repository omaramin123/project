# the library which we use it 
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk 
# The sysmptoms which the module selected issue based it
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

# The disease which module selected from it
disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TRANING DATA  -------------------------------------------------------------------------------------
df=pd.read_csv("c:/Users/DELL/Documents/my study 3th semester/تاريخ التكنولوجيا/tasks/training.csv",sep=",")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# Testing DATA --------------------------------------------------------------------------------
tr=pd.read_csv("c:/Users/DELL/Documents/my study 3th semester/تاريخ التكنولوجيا/tasks/testing.csv",sep=",")

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

# the module of decision tree

def DecisionTree():

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    y_pred=clf3.predict(X_test)
    print(f"Accuracy for Decision tree model {accuracy_score(y_test,y_pred)}")
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),Symptom6.get(),Symptom7.get(),Symptom8.get()]


    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")

# the module of randomforest 
def randomforest():
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),Symptom6.get(),Symptom7.get(),Symptom8.get()]


    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),Symptom6.get(),Symptom7.get(),Symptom8.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")
# gui_stuff------------------------------------------------------------------------------------

root=Tk()

root.title("prediction heat attack model ")
image = Image.open(r"C:\Users\DELL\Desktop\4th semester\graduation project\WhatsApp Image 2024-05-12 at 19.28.25_09eeeeaa.jpg")
image = image.resize((1550, 800), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image)
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# entry variables
Symptom1 = StringVar()
Symptom1.set("enter the first symptom you feel ")
Symptom2 = StringVar()
Symptom2.set("enter the second symptom you feel ")
Symptom3 = StringVar()
Symptom3.set("enter the third symptom you feel ")
Symptom4 = StringVar()
Symptom4.set("enter the fourth symptom you feel ")
Symptom5 = StringVar()
Symptom5.set("enter the fifth symptom you feel ")
Symptom6 = StringVar()
Symptom6.set("enter the sixth symptom you feel ")
Symptom7 = StringVar()
Symptom7.set("enter the seventh symptom you feel ")
Symptom8 = StringVar()
Symptom8.set("enter the eights symptom you feel ")

Name = StringVar()

# Heading
w2 = Label(root, justify=CENTER, text="The prediction of disease for issues by Symptoms what the patient feels ", fg="black", bg="white")
w2.config(font=("Times New Roman", 20))
w2.grid(row=1, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name of the Patient", fg="black", bg="white")
NameLb.grid(row=6, column=0, pady=15, sticky=W)
NameLb.config(font=("Times New Roman",15))


S1Lb = Label(root, text="Symptom 1", fg="black", bg="white")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)
S1Lb.config(font=("Times New Roman",15))

S2Lb = Label(root, text="Symptom 2", fg="black", bg="white")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)
S2Lb.config(font=("Times New Roman",15))

S3Lb = Label(root, text="Symptom 3", fg="black", bg="white")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)
S3Lb.config(font=("Times New Roman",15))

S4Lb = Label(root, text="Symptom 4", fg="black", bg="white")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)
S4Lb.config(font=("Times New Roman",15))

S5Lb = Label(root, text="Symptom 5", fg="black", bg="white")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)
S5Lb.config(font=("Times New Roman",15))

S6Lb = Label(root, text="Symptom 6", fg="black", bg="white")
S6Lb.grid(row=12, column=0, pady=10, sticky=W)
S6Lb.config(font=("Times New Roman",15))

S7Lb = Label(root, text="Symptom 7", fg="black", bg="white")
S7Lb.grid(row=13, column=0, pady=10, sticky=W)
S7Lb.config(font=("Times New Roman",15))

S8Lb = Label(root, text="Symptom 8", fg="black", bg="white")
S8Lb.grid(row=14, column=0, pady=10, sticky=W)
S8Lb.config(font=("Times New Roman",15))


lrLb = Label(root, text="DecisionTree", fg="black", bg="white")
lrLb.grid(row=18, column=0, pady=10,sticky=W)
lrLb.config(font=("Times New Roman",15))

destreeLb = Label(root, text="RandomForest", fg="black", bg="white")
destreeLb.grid(row=19, column=0, pady=10, sticky=W)
destreeLb.config(font=("Times New Roman",15))

ranfLb = Label(root, text="NaiveBayes", fg="black", bg="white")
ranfLb.grid(row=20, column=0, pady=10, sticky=W)
ranfLb.config(font=("Times New Roman",15))

# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

S6En = OptionMenu(root, Symptom6,*OPTIONS)
S6En.grid(row=12, column=1)

S7En = OptionMenu(root, Symptom7,*OPTIONS)
S7En.grid(row=13, column=1)

S8En = OptionMenu(root, Symptom8,*OPTIONS)
S8En.grid(row=14, column=1)

dst = Button(root, text="DecisionTree", command=DecisionTree,bg="white",fg="black")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Randomforest", command=randomforest,bg="white",fg="black")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="white",fg="black")
lr.grid(row=10, column=3,padx=10)

#textfileds
t1 = Text(root, height=1, width=40,bg="white",fg="black")
t1.grid(row=18, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="white",fg="black")
t2.grid(row=19, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="white",fg="black")
t3.grid(row=20, column=1 , padx=10)

root.mainloop()

