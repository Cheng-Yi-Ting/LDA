# k-平均演算法(k-means clustering)
此專案以Python3進行開發，使用scikit-learn以新聞資料統計詞頻，結合LDA主題模型實作的範例。

### K-means Introduction:
LDA是主題模型，可以對一篇文章進行分析，計算它屬於哪個主題的概率，比如一篇文章，裡面好多詞：蘋果、三星、華為、魅族……等等，那麼這篇文章很有可能是手機這個主題。
```
先簡單的說一下LDA的核心思想：
我們認為每一個文檔Doc都是由多個主題Topic組成，而每一個主題Topic由多個詞Word組成。

通過對語料庫D中所有的文檔d進行分詞或者抽詞處理之後，通過模型訓練，我們得到兩個機率矩陣：一是每一個Doc對應K個Topic的機率；二是每一個Topic對應N個詞組成的詞表的機率。

注意由於LDA是基於詞頻統計的，因此一般不用TF-IDF來做文檔特徵

```

![image](https://github.com/Cheng-Yi-Ting/LDA/blob/master/img/topic.png)

