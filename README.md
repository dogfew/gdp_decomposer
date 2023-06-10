# gdp_decomposer
Декомпозитор ВВП на компоненты (Пильник Н.П., Радионов С.А., Станкевич И.П.) 

# Установка

```
todo!
```
# Примеры
```
>>> from decomposer import Decomposer
>>> decomposer = Decomposer(df, branches, n_components=3)  # инициализировать модель
>>> decomposer.fit(multistarts=10, reg=0.1)  # обучить модель с регуляризацией, сделать 10 мультистартов и выбрать лучший
>>> decomposer.fit(multistarts=1, reg=0, x0=decomposer)  # обучить модель без регуляризации, инициализируя текущими значениями
>>> decomposer.summary()
```

# Оригинальная статья 

https://www.hse.ru/data/2022/02/10/1745303367/%D0%A1%D1%82%D0%B0%D0%BD%D0%BA%D0%B5%D0%B2%D0%B8%D1%87_%D1%81%D1%82%D0%B0%D1%82%D1%8C%D1%8F%206.pdf
