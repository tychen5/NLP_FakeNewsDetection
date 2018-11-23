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
4. 假新聞分類、評比
   * 特性、提醒使用者

## Possible Dataset:
* https://www.kaggle.com/c/fake-news/data (title、author、text、true/false；來自爬文的news articles)
* https://github.com/KaiDMML/FakeNewsNet/tree/master/Data (news source, headline, image, body_text, publish_data, etc、包含真假新聞；爬文新聞)
* https://www.kaggle.com/mrisdal/fake-news (uuidUnique identifier,ord_in_thread,authorauthor of story,publisheddate published ,titletitle of the story,texttext of story,languagedata from webhose.io,crawleddate the story was archived,site_urlsite URL from BS detector,countrydata from webhose.io,domain_rankdata from webhose.io,thread_title,spam_scoredata from webhose.io,main_img_urlimage from story,replies_countnumber of replies,participants_countnumber of participants,likesnumber of Facebook likes,commentsnumber of Facebook comments,sharesnumber of Facebook shares,typetype of website (label from BS detector))
* https://github.com/GeorgeMcIntire/fake_real_news_dataset (csv file and contains 1000s of articles tagged as either real or fake)
* https://www.cs.ucsb.edu/~william/data/liar_dataset.zip (假新聞程度分級；UCSB)
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
* 爬文insight ( https://shift.newco.co/2016/11/09/What-I-Discovered-About-Trump-and-Clinton-From-Analyzing-4-Million-Facebook-Posts/ )
* 分析 ( https://towardsdatascience.com/i-trained-fake-news-detection-ai-with-95-accuracy-and-almost-went-crazy-d10589aa57c 、 http://nbviewer.jupyter.org/github/JasonKessler/fakeout/blob/master/Fake%20News%20Analysis.ipynb)

# REF
* 題目參考資料: http://www.im.ntu.edu.tw/~paton/courses.htm
* 2017題目: https://mega.nz/#!xwdEgAjb!FAVoAznYD7bE5rsoXc7isRJUlAbF0m8mamYe2RiCwMM
* 2010題目: https://mega.nz/#!UlNmXQIS!7dZhNx0Cy9-VyjlEI5GUO5zjIgYNJoe9dUAPaCNcowA
