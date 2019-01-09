# IR_final
Information Retrieval and Text Mining project 2018

**題目候選人:**
* 歌詞產生器=>新聞產生器，看文章產生標題
* fake news insight detection EX:假消息的種類、特徵、特性，從文字角度來看跟真的之區別，模板。川普總統大選，分類、真假度(比較IR、ML方法、NN)
* dcard VS ptt 風氣差別。EX:很兇、很和善、用字、情緒氛圍、情感豐富度、有趣的現象(同一件事情不同角度怎樣的面向)
* 透過sandbox log資料，分析惡意程式動態行為、知道可能是哪一個家族的具有什麼樣的特性或是關鍵特徵
* 從文字分析股價、油價、...etc的走勢
* 分析業配文，假的廣告

## 題目中選人: Fake News Analysis and Insight
1. 蒐集各方假新聞dataset
2. 可以從假新聞或真新聞中分析出什麼樣的消息?
   * 用怎樣的方法分析或比較?
3. 假新聞相較於真新聞有怎樣的特徵?
   * 怎麼抓取特徵或關鍵字?
   * 可能用到的情緒字 / 情緒分析
   * 依照詞性去對假真新聞決定可能會有那些常用字。EX:文字雲
   * 語意分析 
4. 假新聞分類、評比
   * 特性、提醒使用者
   
 ### Proposal
 https://docs.google.com/document/d/10-7H9bPJYQRMdOUdugDlWeifdpvoN9twXZGT-m1fhdc/edit?usp=sharing
 
 Deadline: 禮拜天，禮拜一校稿
 
 * 動機
 * 做甚麼
 * solution insight
 * solution regression / classification
 
 **三個方向 Description**

#### 動機目的、要做什麼

動機: 為什麼要做?因為假新聞氾濫、影響閱聽人、帶選舉風向的問題

1. 假新聞的程度
2. 真假新聞之間有什麼區別
3. (假)新聞的種類

- 比較不同方法的performance

**Solution**

1. TF-IDF。給Tagging
2. POS (part-of-speech tagging) EX:openNLP、NLTK => 1.每個不同dataset的詞性常出現哪些字 2. dictionary by overall dataset依詞性要用哪些字
3. Sentiment Analysis EX:TextBlob、
4. feature selection: 關鍵字、類別鑑別力
5. 作者、來源的助益性。每一種類別的差別
6. regression (ML方法、DL方法) / classification (IR方法)

**GOAL**
1. 在相同dictionary大小下: 沒有分詞性情況下跑出來幾分，有詞性的dictionary跑出來幾分 EX:名詞dictionary幾分，動詞正確率幾趴?
2. 前面所做的insight可以跟最後面產生的dict有關連
3. 假新聞的程度、分類，兩者testing dataset互為兩者
4. 時間切三塊或五塊: 選前、選舉正負一個禮拜、選後，主題、用字、情感的變動

**Dataset**
1. 分為十類別(第二個dataset八類、第一個dataset兩類): 第三個dataset的True、mostly Tru放進去第一個dataset的true；第三個dataset的barely-true、false、pants-fire放進去第一個dataset的False
2. 濾除標點符號跟數字、大寫變小寫 ，只留下 content(最長的attribute)、label (假新聞的程度、類別)

三個dataset的text,label合併資料集：https://drive.google.com/file/d/159YVMypQZbOFM_gU6bfPAYl7iGedSbXK/view?usp=sharing

**分工**

目前只看content(最長的attribute)

1. 璨婷: 十個類別的POS、overall dataset的POS
  https://drive.google.com/drive/folders/19aXlSp2WVin6tDm7KQYtA9xh0dsc-uJD?usp=sharing
  
  https://xlsxwriter.readthedocs.io/worksheet.html

2. 沛瑜: 十個類別的長條圖of情緒分析。文獻探討: 詞性、情緒、feature selection、分類、回歸等等套件的論文
3. AMY: 十個類別的文字雲、頻率圖=>做一個overall的，把各類別常見的term的濾掉
4. Leo: 3 kind of feature selecion、tfidf of building overall dictionary

bs類別的東西要拿掉

testing Kaggle: https://www.kaggle.com/c/fake-news/submit

clf好壞結果、reg好壞結果


## Possible Dataset:
* **https://www.kaggle.com/c/fake-news/data (title、author、text、true/false；來自爬文的news articles)** => 
* https://github.com/KaiDMML/FakeNewsNet/tree/master/Data (news source, headline, image, body_text, publish_data, etc、包含真假新聞；爬文新聞)
* **https://www.kaggle.com/mrisdal/fake-news (uuidUnique identifier,ord_in_thread,authorauthor of story,publisheddate published ,titletitle of the story,texttext of story,languagedata from webhose.io,crawleddate the story was archived,site_urlsite URL from BS detector,countrydata from webhose.io,domain_rankdata from webhose.io,thread_title,spam_scoredata from webhose.io,main_img_urlimage from story,replies_countnumber of replies,participants_countnumber of participants,likesnumber of Facebook likes,commentsnumber of Facebook comments,sharesnumber of Facebook shares,typetype of website (label from BS detector))**  https://github.com/bs-detector/bs-detector
* https://github.com/GeorgeMcIntire/fake_real_news_dataset (csv file and contains 1000s of articles tagged as either real or fake)
* **https://www.cs.ucsb.edu/~william/data/liar_dataset.zip (假新聞程度分級；UCSB)(statement、speaker、conext、label、src)**
* https://www.kaggle.com/jruvika/fake-news-detection (URLs,Headline,Body,Label(T/F)；)
* https://www.kaggle.com/c/fake-news-pair-classification-challenge/data (fake news classification)
* https://github.com/JasonKessler/fakeout (完整的project)
* https://github.com/FakeNewsChallenge/fnc-1 (之前辦過的比賽)

* tweets: https://www.nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731
* datasets: https://data.world/datasets/fake-news 、 https://github.com/sumeetkr/AwesomeFakeNews
* preprocess ref: https://www.kaggle.com/rchitic17/fake-news 、 https://www.kaggle.com/michaleczuszek/fake-news-analysis

### 動機
* https://www.ithome.com.tw/news/127214?fbclid=IwAR0oKz7wm0Ub0Kb5FDh9HAvjKX5tgidTtZrFRSY_kVsgQrue5_-K-5iSC-o
* https://www.ithome.com.tw/news/127201?fbclid=IwAR3_vIk3Pdvsem1d_uAWyaiZHUj8C51JLzene9jYOtc50KL31xgEHiHYfLQ

## Possible Goal:
* 協助使用者判斷真假
* 知道假新聞pattern、用字特性、文章特徵
* 新聞分類
* 真假新聞常用的字
* 爬文insight ( https://shift.newco.co/2016/11/09/What-I-Discovered-About-Trump-and-Clinton-From-Analyzing-4-Million-Facebook-Posts/ )
* 分析 ( https://towardsdatascience.com/i-trained-fake-news-detection-ai-with-95-accuracy-and-almost-went-crazy-d10589aa57c 、 http://nbviewer.jupyter.org/github/JasonKessler/fakeout/blob/master/Fake%20News%20Analysis.ipynb)

# REF
* 題目參考資料: http://www.im.ntu.edu.tw/~paton/courses.htm
* 2017題目: https://mega.nz/#!xwdEgAjb!FAVoAznYD7bE5rsoXc7isRJUlAbF0m8mamYe2RiCwMM
* 2010題目: https://mega.nz/#!UlNmXQIS!7dZhNx0Cy9-VyjlEI5GUO5zjIgYNJoe9dUAPaCNcowA
* 文字雲code: https://www.kaggle.com/ngyptr/python-nltk-sentiment-analysis
* TextBlob情感分析: https://nlp.stanford.edu/courses/cs224n/2009/fp/24.pdf (套用NLTK movie_review當作training data)(https://stackoverflow.com/questions/34518570/how-are-sentiment-analysis-computed-in-blob/34519114#34519114)
* NLTK詞性分析(pos_tager): https://explosion.ai/blog/part-of-speech-pos-tagger-in-python (Greedy Averaged Perceptron tagger?)(taining data Sections 00-18 of the Wall Street Journal sections of OntoNotes 5)(https://stackoverflow.com/questions/32016545/how-does-nltk-pos-tag-work)



Datasets for sentiment analysis are available online.[1][2]

The following is a list of a few open source sentiment analysis tools.

* GATE plugins
* SEAS(gsi-upm/SEAS)
* SAGA(gsi-upm/SAGA)
* Stanford Sentiment Analysis Module (Deeply Moving: Deep Learning for Sentiment Analysis)
* LingPipe (Sentiment Analysis Tutorial)
* TextBlob (Tutorial: Quickstart)[3]
* Opinion Finder (OpinionFinder | MPQA)
* Clips pattern.en (pattern.en | CLiPS)


Open Source Dictionary or resources:

* SentiWordNet
* Bing Liu Datasets (Opinion Mining, Sentiment Analysis, Opinion Extraction)
* General Inquirer Dataset (General Inquirer Categories)
* MPQA Opinion Corpus (MPQA Resources)
* WordNet-Affect (WordNet Domains)
* SenticNet
* Emoji Sentiment Ranking
