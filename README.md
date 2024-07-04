import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv('123.csv', delimiter=';', encoding='utf-8')

# Переименование столбцов
df.columns = ['Дата', 'Количество', 'Стоимость_продажи', 'Себестоимость',
              'Валовая_прибыль', 'отступ', 'склад', 'наименование', 'артикул']

# Обработка данных
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y %H:%M')
df['Количество'] = df['Количество'].str.replace(',', '.').astype(float)
df['Стоимость_продажи'] = df['Стоимость_продажи'].str.replace(',', '.').astype(float)
df['Себестоимость'] = df['Себестоимость'].str.replace(',', '.').astype(float)
df['Валовая_прибыль'] = df['Валовая_прибыль'].str.replace(',', '.').astype(float)

# Удаление слова "шт" из наименований
df['наименование'] = df['наименование'].str.replace(', шт', '')

# Создание признаков
df['Год'] = df['Дата'].dt.year
df['Месяц'] = df['Дата'].dt.month
df['День'] = df['Дата'].dt.day
df['День_недели'] = df['Дата'].dt.dayofweek
df['Квартал'] = df['Дата'].dt.quarter
df['Доходность'] = (df['Валовая_прибыль'] / df['Стоимость_продажи'] * 100).round(2)

# Группировка данных по артикулу и месяцу
grouped = df.groupby(['артикул', 'Год', 'Месяц']).agg({
    'Количество': 'sum',
    'Стоимость_продажи': 'sum',
    'Валовая_прибыль': 'sum',
    'Доходность': 'mean',
    'Квартал': 'first'
}).reset_index()

# Создание лагов (предыдущие 3 месяца)
for i in range(1, 4):
    grouped[f'Количество_lag_{i}'] = grouped.groupby('артикул')['Количество'].shift(i)
    grouped[f'Стоимость_продажи_lag_{i}'] = grouped.groupby('артикул')['Стоимость_продажи'].shift(i)
    grouped[f'Валовая_прибыль_lag_{i}'] = grouped.groupby('артикул')['Валовая_прибыль'].shift(i)

# Создание скользящих средних
grouped['MA_3'] = grouped.groupby('артикул')['Количество'].rolling(window=3).mean().reset_index(0,drop=True)
grouped['MA_6'] = grouped.groupby('артикул')['Количество'].rolling(window=6).mean().reset_index(0,drop=True)

# Создание экспоненциальных скользящих средних
grouped['EMA_3'] = grouped.groupby('артикул')['Количество'].ewm(span=3).mean().reset_index(0,drop=True)
grouped['EMA_6'] = grouped.groupby('артикул')['Количество'].ewm(span=6).mean().reset_index(0,drop=True)

# Создание признака тренда
grouped['Trend'] = grouped.groupby('артикул')['Количество'].diff()

# Создание сезонных признаков
grouped['Sin_month'] = np.sin(2 * np.pi * grouped['Месяц']/12)
grouped['Cos_month'] = np.cos(2 * np.pi * grouped['Месяц']/12)

# Удаление строк с NaN значениями
grouped = grouped.dropna()

# Подготовка данных для модели
X = grouped[['Год', 'Месяц', 'Квартал', 'Доходность',
             'Количество_lag_1', 'Количество_lag_2', 'Количество_lag_3',
             'Стоимость_продажи_lag_1', 'Стоимость_продажи_lag_2', 'Стоимость_продажи_lag_3',
             'Валовая_прибыль_lag_1', 'Валовая_прибыль_lag_2', 'Валовая_прибыль_lag_3',
             'MA_3', 'MA_6', 'EMA_3', 'EMA_6', 'Trend', 'Sin_month', 'Cos_month']]
y = grouped['Количество']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Создание и обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Оценка модели
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Percentage Error: {mape}")

# Прогнозирование минимального рекомендуемого месячного остатка
features = grouped[X.columns]
features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
grouped['Мин_рекомендуемый_остаток'] = model.predict(features_scaled)

# Сохранение результатов
result = grouped.groupby('артикул').agg({
    'Мин_рекомендуемый_остаток': 'mean',
    'Валовая_прибыль': 'sum',
    'Стоимость_продажи': 'sum'
}).reset_index()

result['Мин_рекомендуемый_остаток'] = result['Мин_рекомендуемый_остаток'].round().astype(int)
result['Валовая_доходность_%'] = (result['Валовая_прибыль'] / result['Стоимость_продажи'] * 100).round(2)
result['Валовая_доходность_сумма'] = result['Валовая_прибыль'].round(2)
result['Общая_стоимость_продаж'] = result['Стоимость_продажи'].round(2)

# Объединение наименований для каждого артикула
names = df.groupby('артикул')['наименование'].agg(lambda x: ' | '.join(set(x))).reset_index()
result = result.merge(names, on='артикул', how='left')

# Переупорядочивание столбцов
result = result[['артикул', 'наименование', 'Мин_рекомендуемый_остаток',
                 'Валовая_доходность_%', 'Валовая_доходность_сумма', 'Общая_стоимость_продаж']]

result.to_csv('минимальные_рекомендуемые_остатки_запчастей.csv', index=False, encoding='utf-8-sig')
print("Результаты сохранены в файл 'минимальные_рекомендуемые_остатки_запчастей.csv'")

# Вывод первых 10 строк результата
print(result.head(10))
