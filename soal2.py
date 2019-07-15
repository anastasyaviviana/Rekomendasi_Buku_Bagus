#CONTENT BASED FILTERING

import pandas as pd

#dataframe
df_books=pd.read_csv('books.csv')
df_ratings = pd.read_csv('ratings.csv')

def mergecol(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])
df_books['feature'] = df_books.apply(mergecol,axis=1)
# print(df_books.head())

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer=lambda i:i.split(' '),
    analyzer='word')
matrixFeature = model.fit_transform(df_books['feature'])
feature= model.get_feature_names()
# jml_fitur = len(feature)

# cosinus similarity
from sklearn.metrics.pairwise import cosine_similarity
skor = cosine_similarity(matrixFeature)

# input books with rating higher than 2 (3,4 and 5)
andi1 = df_books[df_books['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
andi2 = df_books[df_books['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
andi3 = df_books[df_books['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
andi4 = df_books[df_books['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
andisuka = [andi1,andi2,andi3,andi4]

budi1 = df_books[df_books['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].tolist()[0]-1 
budi2 = df_books[df_books['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1 
budi3 = df_books[df_books['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1 
budisuka = [budi1,budi2,budi3]

ciko1 = df_books[df_books['original_title']=='Robots and Empire']['book_id'].tolist()[0]-1 
cikosuka = [ciko1]

dedi1 = df_books[df_books['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1 
dedi2 = df_books[df_books['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1 
dedi3 = df_books[df_books['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1 
dedisuka = [dedi1,dedi2,dedi3]

ello1 = df_books[df_books['original_title']=='Doctor Sleep']['book_id'].tolist()[0]-1 
ello2 = df_books[df_books['original_title']=='The Story of Doctor Dolittle']['book_id'].tolist()[0]-1 
ello3 = df_books[df_books['title']=='Bridget Jones\'s Diary (Bridget Jones, #1)']['book_id'].tolist()[0]-1 
ellosuka = [ello1,ello2,ello3]

skor_andi1 = list(enumerate(skor[andi1]))
skor_andi2 = list(enumerate(skor[andi2]))
skor_andi3 = list(enumerate(skor[andi3]))
skor_andi4 = list(enumerate(skor[andi4]))
skor_andi = []
for i in skor_andi1:
    skor_andi.append((i[0],(skor_andi1[i[0]][1]+skor_andi2[i[0]][1]+skor_andi3[i[0]][1]+skor_andi4[i[0]][1])/4))

skor_budi1 = list(enumerate(skor[budi1]))
skor_budi2 = list(enumerate(skor[budi2]))
skor_budi3 = list(enumerate(skor[budi3]))
skor_budi = []
for i in skor_budi1:
    skor_budi.append((i[0],(skor_budi1[i[0]][1]+skor_budi2[i[0]][1]+skor_budi3[i[0]][1])/3))

skor_ciko1 = list(enumerate(skor[ciko1]))

skor_dedi1 = list(enumerate(skor[dedi1]))
skor_dedi2 = list(enumerate(skor[dedi2]))
skor_dedi3 = list(enumerate(skor[dedi3]))
skor_dedi = []
for i in skor_dedi1:
    skor_dedi.append((i[0],(skor_dedi1[i[0]][1]+skor_dedi2[i[0]][1]+skor_dedi3[i[0]][1])/3))

skor_ello1 = list(enumerate(skor[ello1]))
skor_ello2 = list(enumerate(skor[ello2]))
skor_ello3 = list(enumerate(skor[ello3]))
skor_ello = []
for i in skor_ello1:
    skor_ello.append((i[0],(skor_ello1[i[0]][1]+skor_ello2[i[0]][1]+skor_ello3[i[0]][1])/3))

#sorted similarity books
sort_andi = sorted(skor_andi, key = lambda i:i[1], reverse =1)
sort_budi = sorted(skor_budi, key = lambda i:i[1], reverse =1)
sort_ciko = sorted(skor_ciko1, key = lambda i:i[1], reverse =1)
sort_dedi = sorted(skor_dedi, key = lambda i:i[1], reverse =1)
sort_ello = sorted(skor_ello, key = lambda i:i[1], reverse =1)

#top 5 similarity books
recommend_andi = []
for i in sort_andi:
    if i[1]>0:
        recommend_andi.append(i)
recommend_budi = []
for i in sort_budi:
    if i[1]>0:
        recommend_budi.append(i)
recommend_ciko = []
for i in sort_ciko:
    if i[1]>0:
        recommend_ciko.append(i)
recommend_dedi = []
for i in sort_dedi:
    if i[1]>0:
        recommend_dedi.append(i)
recommend_ello = []
for i in sort_ello:
    if i[1]>0:
        recommend_ello.append(i)

print('1. Top 5 book recommendation for Andi:')
for i in range(0,5):
    if recommend_andi[i][0] not in andisuka:
        print('-',df_books['original_title'].iloc[recommend_andi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[recommend_andi[i][0]])

print('\n2. Top 5 book recommendation for Budi:')
for i in range(0,5):
    if recommend_budi[i][0] not in budisuka:
        print('-',df_books['original_title'].iloc[recommend_budi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[recommend_budi[i][0]])

print('\n3. Top 5 book recommendation for Ciko:')
for i in range(0,5):
    if recommend_ciko[i][0] not in cikosuka:
        print('-',df_books['original_title'].iloc[recommend_ciko[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[recommend_ciko[i][0]])

print('\n4.Top 5 book recommendation for Dedi:')
for i in range(0,5):
    if recommend_dedi[i][0] not in dedisuka:
        print('-',df_books['original_title'].iloc[recommend_dedi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[recommend_dedi[i][0]])

print('\n5.Top 5 book recommendation for Ello:')
for i in range(0,5):
    if recommend_ello[i][0] not in ellosuka:
        print('-',df_books['title'].iloc[recommend_ello[i][0]]) 
    else:
        i+=5
        print('-',df_books['title'].iloc[recommend_ello[i][0]])