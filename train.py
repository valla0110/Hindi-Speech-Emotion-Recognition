import numpy as np
import warnings
import itertools
import pandas as pd
from mlxtend.evaluate import confusion_matrix
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


try:
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = np.asarray(max(cm)) / 2.
        cm = np.array(cm)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if np.all(cm[i, j] > thresh) else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    
    def train_knn(filepath):
        
        data=pd.read_csv('train_data.csv')
        #print(data)
        #data.drop(['word'],1,inplace=True)

        #feature
        X=np.array(data.drop(['class'],1))

        #class label
        Y=np.array(data['class'])

        x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,Y,test_size=0.3)


        clf=neighbors.KNeighborsClassifier()
        clf.fit(x_train,y_train)

        accuracy=clf.score(x_test,y_test)
        print('accuracy of the classifier = ',accuracy*100)


        happy=['दिव्य','महसूस','दैवीय','परमात्मा','रूप','जीवन','उपस्थिति','प्यार','प्रेम','भावना','हंसमुख','संतुष्ट','प्रसन्न','उन्मादपूर्ण','उत्तेजित','परसन्न','आनंदपूर्ण','आनंदित','उल्लसित','जीवंत','प्रमुदित','मस्त','शांतिपूर्ण','सुहानी','प्रसन्न','रोमांचित','उत्साहित','धन्य है','महाभाग','आनंदमय','ज़िंदादिल','शिकायत नहीं कर सकता','मोहित','टुकड़े करने वाला उपकरण','गरमागरम','सामग्री','खुशनुमा','ख़ुशियां मनानेवाला','ऊंची उड़ान','समलैंगिक','ज़िंदादिल','संतोष','नशे में चूर','विनोदी','हस रहा','रोशनी','अच्छे लग रहे हो','आनंदित','सातवें आसमान पर','क्रियात्मक','दिलेर','चंचल','शानदार','धूप','गुदगुदी','उत्सुकता की स्थिति','ऊपर','अत्यधिक प्रसन्नता की स्थिति']
        sad=['भयानक','दुख','लग','महसूस','बारे','पता','इतना','वास्तव','समय','लोगों','बुरा','उदास','खराब','निराश','व्याकुल','नीचे','गंभीर','उदासी','दुखी','तंग किया','अमित्र','अप्रसन्न','परेशान','हतोत्साहित','असंतुष्ट','छोड़े','निराशाजनक','तप्त','दुर्भाग्य','बदकिस्मत',]
        angry=['महसूस','घबराहट','कड़वा','गड़बड़ी','गुस्सा','लग','आज','बारे','रूप','थोड़ा','कड़वा','शैतान','लाद','अशिष्ट','खीजा हुआ','यहं से चले जाओ','निष्ठाहीन','अत्याचार','नाराज','ईर्ष्या','पागल','गुस्सा','नाराज','द्वेषपूर्ण','परेशान','लालची','क्रोधित','क्षुद्र','चकित','बेरहम','नाराज़','घृणास्पद','शत्रुतापूर्ण','तुच्छ','विला','सर्दी','घृणा करनेवाला']
        fear=['डरा हुआ', 'चपेट में', 'भयभीत', 'उलझन में', 'संदिग्ध', 'हिलाकर रख दिया', 'अजीब', 'दुविधा में पड़ा हुआ', 'उत्तेजित', 'व्यथित', 'व्याकुल', 'भगदड़ का', 'पर शंका', 'uneast', 'चिंतित', 'हिचकते', 'अनिश्चित', 'भयभीत', 'घबराहट', 'अभिभूत', 'असुविधाजनक', 'अनिच्छुक', 'पर हमला किया', 'अरक्षित', 'चौंका', 'संदेहजनक', 'defencive', 'ऊपर मजबूत', 'अजीब', 'परेशान', 'धमकाया', 'डरा हुआ', 'डरावनी', 'बेचेन होना', 'दबाव', 'असुरक्षित', 'आसानी से डरनेवाला', 'धमकी दी', 'डरा हुआ', 'आशंका', 'उलझन में', 'उन्मत्त', 'डर', 'न्युरोटिक', 'चिंतित', 'काल', 'बेताब', 'अत्याचार', 'ढुलमुल', 'भीगी बिल्ली', 'दुविधा में पड़ा हुआ', 'बेचैन', 'अस्थिर', 'हिल', 'शर्मीला', 'अपर्याप्त', 'चिंतित', 'कायर', 'रोमांचित', 'डर लगता', 'मजबूर', 'संकोची', 'पागल', 'डर', 'डरा हुआ', 'चपेट में', 'भयभीत', 'उलझन में', 'संदिग्ध', 'हिलाकर रख दिया', 'अजीब', 'दुविधा में पड़ा हुआ', 'उत्तेजित', 'व्यथित', 'व्याकुल', 'भगदड़ का', 'पर शंका', 'uneast', 'चिंतित', 'हिचकते', 'अनिश्चित', 'भयभीत', 'घबराहट', 'अभिभूत', 'असुविधाजनक', 'अनिच्छुक', 'पर हमला किया', 'अरक्षित', 'चौंका', 'संदेहजनक', 'defencive', 'ऊपर मजबूत', 'अजीब', 'परेशान', 'धमकाया', 'डरा हुआ', 'डरावनी', 'बेचेन होना', 'दबाव', 'असुरक्षित', 'आसानी से डरनेवाला', 'धमकी दी', 'डरा हुआ', 'आशंका', 'उलझन में', 'उन्मत्त', 'डर', 'न्युरोटिक', 'चिंतित', 'काल', 'बेताब', 'अत्याचार', 'ढुलमुल', 'भीगी बिल्ली', 'दुविधा में पड़ा हुआ', 'बेचैन', 'अस्थिर', 'हिल', 'शर्मीला', 'अपर्याप्त', 'चिंतित', 'कायर', 'रोमांचित', 'डर लगता', 'मजबूर', 'संकोची', 'पागल', 'डर']
        surprise=['हे भगवान', 'कमाल', 'कमाल', 'हैरान', 'गजब का', 'अप्रत्याशित घटना', 'धोखा दिया', 'आश्चर्य', 'अवाक', 'अचरज', 'जिज्ञासु', 'प्रभावित किया', 'अप्रत्याशित घटना', 'अजीब', 'ताज्जुब', 'अचरज', 'घबड़ाया हुआ', 'कमाल', 'हैरानी', 'हैरान', 'शानदार', 'अजीब', 'मजेदार', 'आश्चर्यचकित', 'अभिभूत', 'ऊटपटांग', 'ताज्जुब', 'चकित', 'रोमांचित', 'अप्रत्याशित', 'आश्चर्यचकित', 'कमाल', 'अचरज', 'आश्चर्य', 'हैरान', 'चकित', 'हैरानी', 'हे भगवान', 'वाह', 'अप्रत्याशित', 'अप्रत्याशित घटना', 'ताज्जुब', 'शानदार', 'विशाल', 'विस्तृत', 'आलीशान', 'शानदार', 'प्रशस्त', 'ठाठ का', 'शानदार', 'विराजमान', 'उज्ज्वल', 'चमकनेवाला', 'झलकनेवाला', 'कमाल की', 'स्तंभित', 'लकवावाला', 'सुन्न', 'अप्रत्याशित', 'कमाल', 'कमाल', 'हैरान', 'गजब का', 'अप्रत्याशित घटना', 'धोखा दिया', 'आश्चर्य', 'अवाक', 'अचरज', 'जिज्ञासु', 'प्रभावित किया', 'अप्रत्याशित घटना', 'अजीब', 'ताज्जुब', 'अचरज', 'घबड़ाया हुआ', 'कमाल', 'हैरानी', 'हैरान', 'शानदार', 'अजीब', 'मजेदार', 'आश्चर्यचकित', 'अभिभूत', 'ऊटपटांग', 'ताज्जुब', 'हैरान', 'चकित', 'हैरानी', 'हे भगवान', 'वाह', 'अप्रत्याशित', 'अप्रत्याशित घटना', 'ताज्जुब', 'शानदार', 'विशाल', 'विस्तृत', 'आलीशान', 'शानदार', 'प्रशस्त', 'ठाठ का', 'शानदार', 'विराजमान', 'उज्ज्वल', 'हे भगवान', 'कमाल', 'कमाल', 'हैरान', 'गजब का', 'अप्रत्याशित घटना', 'धोखा दिया', 'आश्चर्य', 'अवाक', 'अचरज', 'जिज्ञासु', 'प्रभावित किया', 'अप्रत्याशित घटना', 'अजीब', 'ताज्जुब', 'अचरज', 'घबड़ाया हुआ', 'कमाल', 'हैरानी', 'हैरान', 'शानदार', 'अजीब', 'मजेदार', 'आश्चर्यचकित', 'अभिभूत', 'ऊटपटांग', 'ताज्जुब', 'चकित', 'रोमांचित', 'अप्रत्याशित', 'आश्चर्यचकित', 'कमाल', 'अचरज', 'आश्चर्य', 'हैरान', 'चकित', 'हैरानी', 'हे भगवान', 'वाह', 'अप्रत्याशित', 'अप्रत्याशित घटना', 'ताज्जुब', 'शानदार', 'विशाल', 'विस्तृत', 'आलीशान', 'शानदार', 'प्रशस्त', 'ठाठ का', 'शानदार', 'विराजमान', 'उज्ज्वल', 'चमकनेवाला', 'झलकनेवाला', 'कमाल की', 'स्तंभित', 'लकवावाला', 'सुन्न', 'अप्रत्याशित', 'कमाल', 'कमाल', 'हैरान', 'गजब का', 'अप्रत्याशित घटना', 'धोखा दिया', 'आश्चर्य', 'अवाक', 'अचरज', 'जिज्ञासु', 'प्रभावित किया', 'अप्रत्याशित घटना', 'अजीब', 'ताज्जुब', 'अचरज', 'घबड़ाया हुआ', 'कमाल', 'हैरानी', 'हैरान', 'शानदार', 'अजीब', 'मजेदार', 'आश्चर्यचकित', 'अभिभूत', 'ऊटपटांग', 'ताज्जुब', 'हैरान', 'चकित', 'हैरानी', 'हे भगवान', 'वाह', 'अप्रत्याशित', 'अप्रत्याशित घटना', 'ताज्जुब', 'शानदार', 'विशाल', 'विस्तृत', 'आलीशान', 'शानदार', 'प्रशस्त', 'ठाठ का', 'शानदार', 'विराजमान', 'उज्ज्वल']

        ch=0
        cs=0
        ca=0
        cf=0
        csup=0
        
        #prediction
        test=[]
        test_data=""
        #test_data='मेरी किस्मत खराब है'
        #test_data='मुझे लगता है कि जीवन खुशी और दुख से भरा है'

        with open(filepath, encoding='utf-8') as s:
            for line in s:
                test_data=test_data+line

        print(test_data)
        word_tokens = word_tokenize(test_data)
        for w in word_tokens:
            if w in happy:
                ch=ch+1
            if w in sad:
                cs=cs+1
            if w in angry:
                ca=ca+1
            if w in fear:
                cf=cf+1
            if w in surprise:
                csup=csup+1
        test.append(ch)
        test.append(cs)
        test.append(ca)
        test.append(cf)
        test.append(csup)


        print(test)

        example_mes =np.array(test)
        example_mes = example_mes.reshape(1,-1)

        prediction=clf.predict(example_mes)

        count=0
        for i in range(len(test)):
            if test[i]==0:
                count=count+1

        if count==len(test):
            print('cannot properly classify')

        else :
            if prediction[0]==1:
                print('angry')
            if prediction[0]==2:
                print('fear')
            if prediction[0]==3:
                print('happy')
            if prediction[0]==4:
                print('sad')
            if prediction[0]==5:
                print('surprise')


        cm = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        i=0
        for x in x_test:
            #print(x[0] , " ", x[1])
            out=prediction=clf.predict(x.reshape(1,-1))
            
            if(y_test[i]==1):
                if(out[0]==1):
                    cm[0][0] = cm[0][0]+1;
                elif(out[0]==2):
                    cm[0][1] = cm[0][1] + 1;
                elif(out[0]==3):
                    cm[0][2] = cm[0][2] + 1;
                elif(out[0]==4):
                    cm[0][3] = cm[0][3] + 1;
                elif(out[0]==5):
                    cm[0][4] = cm[0][4] + 1;
            if(y_test[i]==2):
                if(out[0]==1):
                    cm[1][0] = cm[1][0]+1;
                elif(out[0]==2):
                    cm[1][1] = cm[1][1] + 1;
                elif(out[0]==3):
                    cm[1][2] = cm[1][2] + 1;
                elif(out[0]==4):
                    cm[1][3] = cm[1][3] + 1;
                elif(out[0]==5):
                    cm[1][4] = cm[1][4] + 1;
            if(y_test[i]==3):
                if(out[0]==1):
                    cm[2][0] = cm[2][0]+1;
                elif(out[0]==2):
                    cm[2][1] = cm[2][1] + 1;
                elif(out[0]==3):
                    cm[2][2] = cm[2][2] + 1;
                elif(out[0]==4):
                    cm[2][3] = cm[2][3] + 1;
                elif(out[0]==5):
                    cm[2][4] = cm[2][4] + 1;
            if(y_test[i]==4):
                if(out[0]==1):
                    cm[3][0] = cm[3][0]+1;
                elif(out[0]==2):
                    cm[3][1] = cm[3][1] + 1;
                elif(out[0]==3):
                    cm[3][2] = cm[3][2] + 1;
                elif(out[0]==4):
                    cm[3][3] = cm[3][3] + 1;
                elif(out[0]==5):
                    cm[3][4] = cm[3][4] + 1;
            if(y_test[i]==5):
                if(out[0]==1):
                    cm[4][0] = cm[4][0]+1;
                elif(out[0]==2):
                    cm[4][1] = cm[4][1] + 1;
                elif(out[0]==3):
                    cm[4][2] = cm[4][2] + 1;
                elif(out[0]==4):
                    cm[4][3] = cm[4][3] + 1;
                elif(out[0]==5):
                    cm[4][4] = cm[4][4] + 1;
            i=i+1
                    
                
        classes = ("anger","fear","happy","sad","surprise")
        for i in range(5):
            print(cm[i])
        print("============================================================================");
        plt.figure()
        plot_confusion_matrix(cm,classes,title='Confusion matrix : K-NN')
        
        

    def  roc():
        #feature
        data=pd.read_csv('train_data.csv')
        X=np.array(data.drop(['class'],1))
        #class label
        Y=np.array(data['class'])
        n_classes=5 

        tY=label_binarize(Y,classes=[1,2,3,4,5])
        tx_train,tx_test,ty_train,ty_test = cross_validation.train_test_split(X,tY,test_size=0.3)


        clf=neighbors.KNeighborsClassifier()
        y_score=clf.fit(tx_train,ty_train).predict(tx_test)
        print(y_score)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ty_test[:,i],y_score[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Class ')
            plt.legend(loc="lower right")
        

        
    train_knn('data.txt')
    roc()
    plt.show()

except RuntimeWarning:
    print("")
    #do nothing
