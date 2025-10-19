import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib.ticker import FuncFormatter

# Функция для форматирования денежных значений
def format_currency(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.0f}K'
    else:
        return f'{x:.0f}'

# 1. Загрузка и подготовка данных
df = pd.read_csv('src/fin_v2.csv')

# Преобразуем столбец времени в datetime
df['ВРЕМЯ'] = pd.to_datetime(df['ВРЕМЯ'])

# Создаем дополнительные столбцы для анализа
df['ГОД'] = df['ВРЕМЯ'].dt.year
df['МЕСЯЦ'] = df['ВРЕМЯ'].dt.month
df['ДАТА'] = df['ВРЕМЯ'].dt.date
df['ДЕНЬ_НОМЕР'] = (df['ВРЕМЯ'] - df['ВРЕМЯ'].min()).dt.days

print("\nОСНОВНАЯ ИНФОРМАЦИЯ О ДАННЫХ")
print("─" * 59)
print(f"Период данных: с {df['ВРЕМЯ'].min()} по {df['ВРЕМЯ'].max()}")
print(f"Всего записей: {len(df)}")
print(f"Категории: {df['КАТЕГОРИЯ'].unique().tolist()}")
print(f"Общий доход: {df[df['СУММА'] > 0]['СУММА'].sum():,.2f} руб")
print(f"Общий расход: {df[df['СУММА'] < 0]['СУММА'].sum():,.2f} руб")

# 2. График среднемесячного дохода для каждого года
plt.figure(figsize=(12, 8))

# Среднемесячный доход по годам
monthly_income = df[df['СУММА'] > 0].groupby(['ГОД', 'МЕСЯЦ'])['СУММА'].sum().reset_index()
yearly_avg_income = monthly_income.groupby('ГОД')['СУММА'].mean().reset_index()

plt.subplot(2, 2, 1)
plt.bar(yearly_avg_income['ГОД'], yearly_avg_income['СУММА'], color='green', alpha=0.7)
plt.title('Среднемесячный доход по годам', fontsize=12, fontweight='bold')
plt.xlabel('Год')
plt.ylabel('Средний доход, руб')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
plt.grid(True, alpha=0.3)

# 3. График среднемесячного расхода для каждого года
monthly_expense = df[df['СУММА'] < 0].groupby(['ГОД', 'МЕСЯЦ'])['СУММА'].sum().reset_index()
monthly_expense['СУММА'] = monthly_expense['СУММА'].abs()  # Преобразуем в положительные значения
yearly_avg_expense = monthly_expense.groupby('ГОД')['СУММА'].mean().reset_index()

plt.subplot(2, 2, 2)
plt.bar(yearly_avg_expense['ГОД'], yearly_avg_expense['СУММА'], color='red', alpha=0.7)
plt.title('Среднемесячный расход по годам', fontsize=12, fontweight='bold')
plt.xlabel('Год')
plt.ylabel('Средний расход, руб')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
plt.grid(True, alpha=0.3)

# 4. Состояние счета на каждый день
daily_balance = df.groupby('ДАТА')['СУММА'].sum().reset_index()
daily_balance['СОСТОЯНИЕ_СЧЕТА'] = daily_balance['СУММА'].cumsum()
daily_balance = daily_balance.sort_values('ДАТА')

# Преобразуем ДАТА обратно в datetime для вычислений
daily_balance['ДАТА_DT'] = pd.to_datetime(daily_balance['ДАТА'])
daily_balance['ДЕНЬ_НОМЕР'] = (daily_balance['ДАТА_DT'] - daily_balance['ДАТА_DT'].min()).dt.days

plt.subplot(2, 2, 3)
plt.plot(daily_balance['ДАТА'], daily_balance['СОСТОЯНИЕ_СЧЕТА'], linewidth=2)
plt.title('Состояние счета по дням', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Состояние счета, руб')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 5. Анализ отрицательного счета
print("\n" + "="*50)
print("          1. АНАЛИЗ ОТРИЦАТЕЛЬНОГО СЧЕТА")
print("="*50)

negative_balance = daily_balance[daily_balance['СОСТОЯНИЕ_СЧЕТА'] < 0]
if not negative_balance.empty:
    first_negative = negative_balance['ДАТА'].min()
    last_negative = negative_balance['ДАТА'].max()
    print(f"Первая дата с отрицательным счетом: {first_negative}")
    print(f"Последняя дата с отрицательным счетом: {last_negative}")
    print(f"Количество дней с отрицательным счетом: {len(negative_balance)}")
else:
    print("Отрицательного счета не было")

# 6. Анализ крупных трат
print("\n" + "="*50)
print("          2. АНАЛИЗ КРУПНЫХ ТРАТ")
print("="*50)

biggest_expenses = df[df['СУММА'] < 0].nsmallest(5, 'СУММА')[['ВРЕМЯ', 'СУММА', 'КАТЕГОРИЯ']]
biggest_expenses['СУММА'] = biggest_expenses['СУММА'].abs()

print("ТОП-5 САМЫХ БОЛЬШИХ ТРАТ:")
print("─" * 60)
for i, row in biggest_expenses.iterrows():
    print(f"{i+1:2d}. {row['ВРЕМЯ'].strftime('%Y-%m-%d %H:%M')}")
    print(f"    Сумма: {row['СУММА']:,.0f} руб | Категория: {row['КАТЕГОРИЯ']}")

# 7. Прогноз состояния счета с использованием numpy
def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def polynomial_regression(x, y, degree=2):
    # Создаем матрицу признаков для полинома
    A = np.vander(x, degree + 1)
    # Решаем систему уравнений
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return coeffs

def predict_polynomial(x, coeffs):
    return np.polyval(coeffs, x)

# Подготовка данных для прогноза
balance_data = daily_balance[['ДАТА', 'СОСТОЯНИЕ_СЧЕТА', 'ДЕНЬ_НОМЕР']].copy()

X = balance_data['ДЕНЬ_НОМЕР'].values
y = balance_data['СОСТОЯНИЕ_СЧЕТА'].values

# Линейная регрессия
m_lin, c_lin = linear_regression(X, y)
y_lin_pred = m_lin * X + c_lin

# Квадратичная регрессия
poly_coeffs = polynomial_regression(X, y, degree=2)
y_poly_pred = predict_polynomial(X, poly_coeffs)

# Прогноз на год вперед
last_date = balance_data['ДАТА'].iloc[-1]
last_date_dt = pd.to_datetime(last_date)
days_ahead = 365
future_days = np.array(range(balance_data['ДЕНЬ_НОМЕР'].max() + 1,
                           balance_data['ДЕНЬ_НОМЕР'].max() + 1 + days_ahead))

future_lin_pred = m_lin * future_days + c_lin
future_poly_pred = predict_polynomial(future_days, poly_coeffs)

future_dates = [last_date_dt + timedelta(days=i) for i in range(1, days_ahead + 1)]

plt.subplot(2, 2, 4)
plt.plot(balance_data['ДАТА'], balance_data['СОСТОЯНИЕ_СЧЕТА'], label='Фактические данные', linewidth=2)
plt.plot(balance_data['ДАТА'], y_lin_pred, '--', label='Линейная модель', alpha=0.7)
plt.plot(balance_data['ДАТА'], y_poly_pred, '--', label='Квадратичная модель', alpha=0.7)
plt.plot(future_dates, future_lin_pred, ':', label='Линейный прогноз', alpha=0.7)
plt.plot(future_dates, future_poly_pred, ':', label='Квадратичный прогноз', alpha=0.7)
plt.title('Прогноз состояния счета', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Состояние счета, руб')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("          3. ПРОГНОЗ СОСТОЯНИЯ СЧЕТА")
print("="*50)

print(f"Прогноз на год вперед (последняя дата: {last_date}):")
print("─" * 50)
print(f"   Линейный прогноз через год: {future_lin_pred[-1]:,.0f} руб")
print(f"   Квадратичный прогноз через год: {future_poly_pred[-1]:,.0f} руб")

# 8. Прогноз без двух самых больших трат
print("\n" + "="*50)
print("          4. ПРОГНОЗ БЕЗ ВЫБРОСОВ")
print("="*50)

df_cleaned = df.copy()
biggest_expense_indices = df[df['СУММА'] < 0].nsmallest(2, 'СУММА').index
df_cleaned = df_cleaned.drop(biggest_expense_indices)

# Пересчитываем состояние счета без выбросов
daily_balance_cleaned = df_cleaned.groupby('ДАТА')['СУММА'].sum().reset_index()
daily_balance_cleaned['СОСТОЯНИЕ_СЧЕТА'] = daily_balance_cleaned['СУММА'].cumsum()
daily_balance_cleaned = daily_balance_cleaned.sort_values('ДАТА')

# Преобразуем ДАТА обратно в datetime для вычислений
daily_balance_cleaned['ДАТА_DT'] = pd.to_datetime(daily_balance_cleaned['ДАТА'])
daily_balance_cleaned['ДЕНЬ_НОМЕР'] = (daily_balance_cleaned['ДАТА_DT'] - daily_balance_cleaned['ДАТА_DT'].min()).dt.days

balance_data_cleaned = daily_balance_cleaned[['ДАТА', 'СОСТОЯНИЕ_СЧЕТА', 'ДЕНЬ_НОМЕР']].copy()

X_clean = balance_data_cleaned['ДЕНЬ_НОМЕР'].values
y_clean = balance_data_cleaned['СОСТОЯНИЕ_СЧЕТА'].values

# Обучение моделей на очищенных данных
m_lin_clean, c_lin_clean = linear_regression(X_clean, y_clean)
y_lin_pred_clean = m_lin_clean * X_clean + c_lin_clean

poly_coeffs_clean = polynomial_regression(X_clean, y_clean, degree=2)
y_poly_pred_clean = predict_polynomial(X_clean, poly_coeffs_clean)

# Прогноз на год вперед для очищенных данных
future_lin_pred_clean = m_lin_clean * future_days + c_lin_clean
future_poly_pred_clean = predict_polynomial(future_days, poly_coeffs_clean)

print("ПРОГНОЗ БЕЗ ДВУХ САМЫХ БОЛЬШИХ ТРАТ:")
print("─" * 50)
print(f"   Линейный прогноз через год: {future_lin_pred_clean[-1]:,.0f} руб")
print(f"   Квадратичный прогноз через год: {future_poly_pred_clean[-1]:,.0f} руб")
print(f"   Разница в прогнозах: {future_poly_pred_clean[-1] - future_poly_pred[-1]:,.0f} руб")

# Визуализация прогнозов без выбросов
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(balance_data_cleaned['ДАТА'], balance_data_cleaned['СОСТОЯНИЕ_СЧЕТА'],
         label='Фактические данные (без выбросов)', linewidth=2)
plt.plot(balance_data_cleaned['ДАТА'], y_lin_pred_clean, '--', label='Линейная модель', alpha=0.7)
plt.plot(balance_data_cleaned['ДАТА'], y_poly_pred_clean, '--', label='Квадратичная модель', alpha=0.7)
plt.plot(future_dates, future_lin_pred_clean, ':', label='Линейный прогноз', alpha=0.7)
plt.plot(future_dates, future_poly_pred_clean, ':', label='Квадратичный прогноз', alpha=0.7)
plt.title('Прогноз без двух самых больших трат', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Состояние счета, руб')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Сравнение прогнозов
plt.subplot(1, 2, 2)
comparison_dates = future_dates[-30:]  # Последние 30 дней для наглядности
plt.plot(comparison_dates, future_lin_pred[-30:], label='Линейный (с выбросами)', linewidth=2)
plt.plot(comparison_dates, future_poly_pred[-30:], label='Квадратичный (с выбросами)', linewidth=2)
plt.plot(comparison_dates, future_lin_pred_clean[-30:], '--', label='Линейный (без выбросов)', linewidth=2)
plt.plot(comparison_dates, future_poly_pred_clean[-30:], '--', label='Квадратичный (без выбросов)', linewidth=2)
plt.title('Сравнение прогнозов (последние 30 дней)', fontsize=12, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Состояние счета, руб')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 9. Дополнительная аналитика
print("\n" + "="*50)
print("          5. ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА")
print("="*50)

print("ОБЩАЯ СТАТИСТИКА:")
print("─" * 50)
print(f"   Всего транзакций: {len(df)}")
print(f"   Доходных операций: {len(df[df['СУММА'] > 0])}")
print(f"   Расходных операций: {len(df[df['СУММА'] < 0])}")
print(f"   Средняя сумма дохода: {df[df['СУММА'] > 0]['СУММА'].mean():,.0f} руб")
print(f"   Средняя сумма расхода: {df[df['СУММА'] < 0]['СУММА'].mean():,.0f} руб")

print("\nСТАТИСТИКА ПО КАТЕГОРИЯМ РАСХОДОВ:")
print("─" * 52)

expense_by_category = df[df['СУММА'] < 0].groupby('КАТЕГОРИЯ')['СУММА'].agg(['count', 'sum']).abs()
expense_by_category = expense_by_category.sort_values('sum', ascending=False)
expense_by_category.columns = ['Количество операций', 'Общая сумма']

# Форматируем таблицу
category_str = expense_by_category.head().to_string(
    float_format=lambda x: f'{x:,.0f}',
    justify='center',
    col_space=15
)
lines = category_str.split('\n')
print(lines[0])  # заголовок
print('─' * len(lines[0]))
for line in lines[1:]:
    print(line)

# Ключевые выводы
print("\n" + "="*50)
print("          КЛЮЧЕВЫЕ ВЫВОДЫ")
print("="*50)

# Находим самую затратную категорию
most_expensive_category = expense_by_category.index[0]
most_expensive_amount = expense_by_category.iloc[0]['Общая сумма']

# Анализ доходов/расходов
total_income = df[df['СУММА'] > 0]['СУММА'].sum()
total_expense = df[df['СУММА'] < 0]['СУММА'].sum()
balance_ratio = abs(total_income / total_expense) if total_expense != 0 else 0

print("ФИНАНСОВЫЕ ПОКАЗАТЕЛИ:")
print("─" * 50)
print(f"   Общий доход: {total_income:,.0f} руб")
print(f"   Общий расход: {abs(total_expense):,.0f} руб")
print(f"   Соотношение доход/расход: {balance_ratio:.2f}")

print("\nАНАЛИТИЧЕСКИЕ ВЫВОДЫ:")
print("─" * 52)
print(f"   Самая затратная категория: {most_expensive_category}")
print(f"   Сумма по самой затратной категории: {most_expensive_amount:,.0f} руб")

if future_poly_pred[-1] > future_poly_pred_clean[-1]:
    print("   Прогноз улучшился после исключения выбросов")
else:
    print("   Прогноз ухудшился после исключения выбросов")
