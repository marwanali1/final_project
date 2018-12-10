import sys
import numpy as np
from scipy.sparse import *
import pickle
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from scipy.sparse import *
from sklearn.svm import *
from sklearn.model_selection import *
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions as scikit
import os

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
    
def read_in_top_genres(): #load in the mapping for tracks belonging to ONLY the top 13 genres 
    trackID_Genre = {}
    with open("../data/lyric_data/ancillary_data/msd-topMAGD-genreAssignment.txt", 'r') as input:
        for i,line in enumerate(input):
            line = line.strip('\n')
            temp = line.split('\t')
            trackID_Genre[temp[0]] = temp[1]
            print("Processing Genre Data Item #{}".format(i+1))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
    return trackID_Genre
    
def load_data(reloadGenreMap,reloadTrainData,reloadtestData):
        train_labels = []
        train_validTracks = []
        train_validTrackIDs = []
        genremap = None
        
        if(reloadTrainData):
            if(not os.path.exists("../data/lyric_data/ancillary_data/train_raw.dat")):
                print("\nYou must unzip the raw training file in order to regenerate the train data. Place this unzipped file back into '../data/lyric_data/ancillary_data/' \nThis was zipped to circumvent GitHub's file upload size limit.\nThe zipped data file is located inside ../data/lyric_data/ancillary_data/train_raw.zip\n")
                exit(0)
            else:    
                print("Regenerating Pickled Training Data")
                trackIDs = []
                lyrics = []
                with open("../data/lyric_data/ancillary_data/train_raw.dat", 'r') as input:
                    for i,line in enumerate(input):
                        line = line.strip('\n')
                        temp = line.split(",")
                        words = temp[2:]
                        vector = [0]*5000 #dok_matrix((1,5000),dtype=np.int8)
                        for word in words:
                            item = word.split(":")
                            vector[int(item[0])-1] = int(item[1])
                        lyrics.append(lil_matrix([vector]))
                        trackIDs.append(temp[0])
                        print("Reading Raw Train Data Item #{}".format(i+1))
                        sys.stdout.write(CURSOR_UP_ONE)
                        sys.stdout.write(ERASE_LINE)
            if(reloadGenreMap):
                print("Regenerating Genre Map.")
                genremap = read_in_top_genres()
                pickle.dump(genremap,open("../data/lyric_data/ancillary_data/genremap.dat",'wb'), protocol=2)
            else:
                print("Pregenerated Genre Map Found.")
                genremap = pickle.load(open("../data/lyric_data/ancillary_data/genremap.dat",'rb'))
                print("Genre Map Loaded.")
                
            poprocktracksadded = 0
            for i,track in enumerate(trackIDs):
                g = ["Country","Electronic","Folk","International","Pop_Rock","Rap","RnB","Vocal","Latin"]
                try:
                    if (genremap[track] in g):
                        if(genremap[track] == "Pop_Rock"):
                            if (genremap[track] == "Pop_Rock" and poprocktracksadded < 5753):
                                poprocktracksadded += 1
                                print("Checking for track #{}.".format(i+1))
                                train_labels.append(genremap[track])
                                train_validTracks.append(lyrics[i])
                                train_validTrackIDs.append(track)
                                print("Track found in mapping.")
                            else:
                                continue
                        else:
                            print("Checking for track #{}.".format(i+1))
                            train_labels.append(genremap[track])
                            train_validTracks.append(lyrics[i])
                            train_validTrackIDs.append(track)
                            print("Track found in mapping.")
                except:
                    print("Track not found in mapping.")
            pickle.dump(train_labels,open("../data/lyric_data/train_labels.dat",'wb'), protocol=2)
            pickle.dump(train_validTracks,open("../data/lyric_data/train_tracks.dat",'wb'), protocol=2)
            pickle.dump(train_validTrackIDs,open("../data/lyric_data/ancillary_data/train_track_ids.dat",'wb'), protocol=2)
        else:
            print("Pickled Training Data File Found.")
            train_labels = pickle.load(open("../data/lyric_data/train_labels.dat",'rb'))
            train_validTracks = pickle.load(open("../data/lyric_data/train_tracks.dat",'rb'))
            train_validTrackIDs = pickle.load(open("../data/lyric_data/ancillary_data/train_track_ids.dat",'rb'))
            print("Training Data Loaded.")
            
        test_labels = []
        test_validTracks = []
        test_validTrackIDs = []
        country = 0
        electronic = 0
        folk = 0
        international = 0
        latin = 0
        poprock = 0
        rap = 0
        rnb = 0
        vocal = 0
        gnames = ["Country","Electronic","Folk","International","Pop_Rock","Rap","RnB","Vocal","Latin"]
        g = {"Country":0,"Electronic":0,"Folk":0,"International":0,"Pop_Rock":0,"Rap":0,"RnB":0,"Vocal":0,"Latin":0}
        if(reloadtestData):
            print("Reloading Test Data")
            for indx,id in enumerate(train_validTrackIDs):
                if(g[genremap[id]] < 300):
                    g[genremap[id]] += 1
                    test_labels.append(genremap[id])
                    test_validTracks.append(train_validTracks[indx])
                    test_validTrackIDs.append(train_validTrackIDs[indx])
                    del train_labels[indx]
                    del train_validTracks[indx]
                    del train_validTrackIDs[indx]
            pickle.dump(test_labels,open("../data/lyric_data/test_labels.dat",'wb'), protocol=2)
            pickle.dump(test_validTracks,open("../data/lyric_data/test_tracks.dat",'wb'), protocol=2)
            pickle.dump(test_validTrackIDs,open("../data/lyric_data/ancillary_data/test_track_ids.dat",'wb'), protocol=2)
        else:
            print("Pickled Testing Data File Found.")
            test_labels = pickle.load(open("../data/lyric_data/test_labels.dat",'rb'))
            test_validTracks = pickle.load(open("../data/lyric_data/test_tracks.dat",'rb'))
            test_validTrackIDs = pickle.load(open("../data/lyric_data/ancillary_data/test_track_ids.dat",'rb'))
            print("Testing Data Loaded.")
        return train_labels,train_validTracks,train_validTrackIDs,test_labels,test_validTracks,test_validTrackIDs

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=scikit.UndefinedMetricWarning)
    
    knn = KNeighborsClassifier(n_neighbors = 13, algorithm = 'ball_tree',n_jobs = -1)
    linearSVC = LinearSVC(C= 10.0, verbose = 20, random_state = 484, max_iter = 5000)
    complementNB = ComplementNB(alpha = 1)
    dt = DecisionTreeClassifier(max_depth = 10, max_features = 1000,random_state = 484)
    
    classifier = None
    try:
        if(sys.argv[1] == "knn"):
            classifier = knn
        elif(sys.argv[1] == "svm"):
            classifier = linearSVC
        elif(sys.argv[1] == "nb"):
            classifier = complementNB
        elif(sys.argv[1] == "dt"):
            classifier = dt
        else:
            classifier = nb
    except:    
            print("\nYou must specify a valid classifier. \n('knn' = K Nearest Neighbor)\n('svm' = Linear Support Vector Machine)\n('nb' = Naive Bayes)\n('dt' = Decision Tree)")
            exit(0)
    
    reloadTrainData = True
    reloadtestData = True
    reloadGenreMap = True
    
    if(os.path.exists("../data/lyric_data/test_labels.dat") and os.path.exists("../data/lyric_data/test_tracks.dat") and os.path.exists("../data/lyric_data/ancillary_data/test_track_ids.dat")):
        reloadtestData = False
    
    if(os.path.exists("../data/lyric_data/train_labels.dat") and os.path.exists("../data/lyric_data/train_tracks.dat") and os.path.exists("../data/lyric_data/ancillary_data/train_track_ids.dat")):
        reloadTrainData = False
    
    if(os.path.exists("../data/lyric_data/ancillary_data/genremap.dat")):
        reloadGenreMap = False
    
    writeout_start = "===== " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " =====\n"    
    train_labels,train_tracks,train_track_ids,test_labels,test_tracks,test_track_ids = load_data(reloadGenreMap,reloadTrainData,reloadtestData)
    
    '''
    f1_function = make_scorer(f1_score,greater_is_better=True,average='micro')
    parameters = {'n_neighbors': range(3,15,2)}#{'C': range(10,20,10), 'max_iter': range(10000,20000,1000)}#{'alpha' : range(1,101,5)}
    clf = GridSearchCV(estimator=classifier,param_grid=parameters,cv=5,scoring =f1_function, refit = True, n_jobs=-1,verbose=40)
    clf.fit(vstack(train_tracks),train_labels)
    print("best {}|   Best Score {}".format(clf.best_params_,clf.best_score_))
    '''
    
    classifier.fit(vstack(train_tracks), train_labels)
    
    predictions = classifier.predict(vstack(test_tracks))
    
    print("\nClassifier: {}\n".format(sys.argv[1]))
    print(writeout_start)
    print(classification_report(test_labels,predictions, target_names = classifier.classes_))
    
    writeout_end = "===== " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " =====\n"
    print(writeout_end)
    
