
# CRISP-DM Metodolojisi (Cross-Industry Standard Process for Data Science)

[crisp-dm-metodolojisi-nedir](http://http://www.leylatilki.com/crisp-dm-metodolojisi-nedir/)

[crisp-dm-hiyerarsik-surec-modeli](http://www.leylatilki.com/crisp-dm-hiyerarsik-surec-modeli/)

## İçerik

* [1. Veri Yükleme](#1.-Veri-Yükleme)
* [2. Veri Ön İşleme](#2.-Veri-Ön-İşleme)
    * [a. Eksik Veriler](#a.-Eksik-Veriler)
    * [b. Kategorik Veriler](#b.-Kategorik-Veriler)
    * [c. Verilerin Birleştirilmesi](#c.-Verilerin-Birleştirilmesi)

## 1. Veri Yükleme

[pandas.DataFrame.to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html)


```python
import pandas as pd

# csv formatındaki verisetini yüklemek için:
veriler = pd.read_csv('veriler/eksikveriler.csv')

# İlk 5 satırı ekrana yazdırmak için;
print(veriler)
```

       ulke  boy  kilo   yas cinsiyet
    0    tr  130    30  10.0        e
    1    tr  125    36  11.0        e
    2    tr  135    34  10.0        k
    3    tr  133    30   9.0        k
    4    tr  129    38  12.0        e
    5    tr  180    90  30.0        e
    6    tr  190    80  25.0        e
    7    tr  175    90  35.0        e
    8    tr  177    60  22.0        k
    9    us  185   105  33.0        e
    10   us  165    55  27.0        k
    11   us  155    50  44.0        k
    12   us  160    58   NaN        k
    13   us  162    59  41.0        k
    14   us  167    62  55.0        k
    15   fr  174    70  47.0        e
    16   fr  193    90   NaN        e
    17   fr  187    80  27.0        e
    18   fr  183    88  28.0        e
    19   fr  159    40  29.0        k
    20   fr  164    66  32.0        k
    21   fr  166    56  42.0        k


## 2. Veri Ön İşleme

* **Veri**
    * **Kategorik**
        * Nominal
            * *Binominal*
            * *Polinominal*
        * Ordinal
    * **Sayısal**
        * Oransal
        * Aralık

### a. Eksik Veriler

[sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)


```python
sayisal_veriler = veriler.iloc[:,1:4].values

print(sayisal_veriler)
```

    [[130.  30.  10.]
     [125.  36.  11.]
     [135.  34.  10.]
     [133.  30.   9.]
     [129.  38.  12.]
     [180.  90.  30.]
     [190.  80.  25.]
     [175.  90.  35.]
     [177.  60.  22.]
     [185. 105.  33.]
     [165.  55.  27.]
     [155.  50.  44.]
     [160.  58.  nan]
     [162.  59.  41.]
     [167.  62.  55.]
     [174.  70.  47.]
     [193.  90.  nan]
     [187.  80.  27.]
     [183.  88.  28.]
     [159.  40.  29.]
     [164.  66.  32.]
     [166.  56.  42.]]



```python
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(sayisal_veriler[:,1:4])

sayisal_veriler[:,1:4] = imputer.transform(sayisal_veriler[:,1:4])

print(sayisal_veriler)
```

    [[130.    30.    10.  ]
     [125.    36.    11.  ]
     [135.    34.    10.  ]
     [133.    30.     9.  ]
     [129.    38.    12.  ]
     [180.    90.    30.  ]
     [190.    80.    25.  ]
     [175.    90.    35.  ]
     [177.    60.    22.  ]
     [185.   105.    33.  ]
     [165.    55.    27.  ]
     [155.    50.    44.  ]
     [160.    58.    28.45]
     [162.    59.    41.  ]
     [167.    62.    55.  ]
     [174.    70.    47.  ]
     [193.    90.    28.45]
     [187.    80.    27.  ]
     [183.    88.    28.  ]
     [159.    40.    29.  ]
     [164.    66.    32.  ]
     [166.    56.    42.  ]]


### b. Kategorik Veriler

[label-encoder-vs-one-hot-encoder](http://www.leylatilki.com/label-encoder-vs-one-hot-encoder/)


```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

ulkeler = veriler.iloc[:,0:1].values

ulkeler[:,0] = le.fit_transform(ulkeler[:,0])

print(ulkeler)
```

    [[1]
     [1]
     [1]
     [1]
     [1]
     [1]
     [1]
     [1]
     [1]
     [2]
     [2]
     [2]
     [2]
     [2]
     [2]
     [0]
     [0]
     [0]
     [0]
     [0]
     [0]
     [0]]



```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

ulkeler = veriler.iloc[:,0:1].values

ulkeler = ohe.fit_transform(ulkeler).toarray()

print(ulkeler)
```

    [[0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 0. 1.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]]


### c. Verilerin Birleştirilmesi

[pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

[pandas.concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html)


```python
ulkeler = pd.DataFrame(data = ulkeler, index=range(22), columns=['fr', 'tr', 'us'])

print(ulkeler)
```

         fr   tr   us
    0   0.0  1.0  0.0
    1   0.0  1.0  0.0
    2   0.0  1.0  0.0
    3   0.0  1.0  0.0
    4   0.0  1.0  0.0
    5   0.0  1.0  0.0
    6   0.0  1.0  0.0
    7   0.0  1.0  0.0
    8   0.0  1.0  0.0
    9   0.0  0.0  1.0
    10  0.0  0.0  1.0
    11  0.0  0.0  1.0
    12  0.0  0.0  1.0
    13  0.0  0.0  1.0
    14  0.0  0.0  1.0
    15  1.0  0.0  0.0
    16  1.0  0.0  0.0
    17  1.0  0.0  0.0
    18  1.0  0.0  0.0
    19  1.0  0.0  0.0
    20  1.0  0.0  0.0
    21  1.0  0.0  0.0



```python
sayisal_veriler = pd.DataFrame(data = sayisal_veriler, index = range(22), columns = ['boy', 'kilo', 'yas'])
print(sayisal_veriler)
```

          boy   kilo    yas
    0   130.0   30.0  10.00
    1   125.0   36.0  11.00
    2   135.0   34.0  10.00
    3   133.0   30.0   9.00
    4   129.0   38.0  12.00
    5   180.0   90.0  30.00
    6   190.0   80.0  25.00
    7   175.0   90.0  35.00
    8   177.0   60.0  22.00
    9   185.0  105.0  33.00
    10  165.0   55.0  27.00
    11  155.0   50.0  44.00
    12  160.0   58.0  28.45
    13  162.0   59.0  41.00
    14  167.0   62.0  55.00
    15  174.0   70.0  47.00
    16  193.0   90.0  28.45
    17  187.0   80.0  27.00
    18  183.0   88.0  28.00
    19  159.0   40.0  29.00
    20  164.0   66.0  32.00
    21  166.0   56.0  42.00



```python
cinsiyetler = veriler.iloc[:,-1:].values

cinsiyetler = pd.DataFrame(data = cinsiyetler, index = range(22), columns=['cinsiyet'])

print(cinsiyetler)
```

       cinsiyet
    0         e
    1         e
    2         k
    3         k
    4         e
    5         e
    6         e
    7         e
    8         k
    9         e
    10        k
    11        k
    12        k
    13        k
    14        k
    15        e
    16        e
    17        e
    18        e
    19        k
    20        k
    21        k



```python
birlestirilmis_veri = pd.concat([ulkeler, sayisal_veriler], axis=1)

print(birlestirilmis_veri)
```

         fr   tr   us    boy   kilo    yas
    0   0.0  1.0  0.0  130.0   30.0  10.00
    1   0.0  1.0  0.0  125.0   36.0  11.00
    2   0.0  1.0  0.0  135.0   34.0  10.00
    3   0.0  1.0  0.0  133.0   30.0   9.00
    4   0.0  1.0  0.0  129.0   38.0  12.00
    5   0.0  1.0  0.0  180.0   90.0  30.00
    6   0.0  1.0  0.0  190.0   80.0  25.00
    7   0.0  1.0  0.0  175.0   90.0  35.00
    8   0.0  1.0  0.0  177.0   60.0  22.00
    9   0.0  0.0  1.0  185.0  105.0  33.00
    10  0.0  0.0  1.0  165.0   55.0  27.00
    11  0.0  0.0  1.0  155.0   50.0  44.00
    12  0.0  0.0  1.0  160.0   58.0  28.45
    13  0.0  0.0  1.0  162.0   59.0  41.00
    14  0.0  0.0  1.0  167.0   62.0  55.00
    15  1.0  0.0  0.0  174.0   70.0  47.00
    16  1.0  0.0  0.0  193.0   90.0  28.45
    17  1.0  0.0  0.0  187.0   80.0  27.00
    18  1.0  0.0  0.0  183.0   88.0  28.00
    19  1.0  0.0  0.0  159.0   40.0  29.00
    20  1.0  0.0  0.0  164.0   66.0  32.00
    21  1.0  0.0  0.0  166.0   56.0  42.00



```python
birlestirilmis_veri = pd.concat([birlestirilmis_veri, cinsiyetler], axis=1)

print(birlestirilmis_veri)
```

         fr   tr   us    boy   kilo    yas cinsiyet
    0   0.0  1.0  0.0  130.0   30.0  10.00        e
    1   0.0  1.0  0.0  125.0   36.0  11.00        e
    2   0.0  1.0  0.0  135.0   34.0  10.00        k
    3   0.0  1.0  0.0  133.0   30.0   9.00        k
    4   0.0  1.0  0.0  129.0   38.0  12.00        e
    5   0.0  1.0  0.0  180.0   90.0  30.00        e
    6   0.0  1.0  0.0  190.0   80.0  25.00        e
    7   0.0  1.0  0.0  175.0   90.0  35.00        e
    8   0.0  1.0  0.0  177.0   60.0  22.00        k
    9   0.0  0.0  1.0  185.0  105.0  33.00        e
    10  0.0  0.0  1.0  165.0   55.0  27.00        k
    11  0.0  0.0  1.0  155.0   50.0  44.00        k
    12  0.0  0.0  1.0  160.0   58.0  28.45        k
    13  0.0  0.0  1.0  162.0   59.0  41.00        k
    14  0.0  0.0  1.0  167.0   62.0  55.00        k
    15  1.0  0.0  0.0  174.0   70.0  47.00        e
    16  1.0  0.0  0.0  193.0   90.0  28.45        e
    17  1.0  0.0  0.0  187.0   80.0  27.00        e
    18  1.0  0.0  0.0  183.0   88.0  28.00        e
    19  1.0  0.0  0.0  159.0   40.0  29.00        k
    20  1.0  0.0  0.0  164.0   66.0  32.00        k
    21  1.0  0.0  0.0  166.0   56.0  42.00        k



```python

```
