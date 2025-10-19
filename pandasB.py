import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Загрузка данных
students = pd.read_csv('src/students.csv')
grades = pd.read_csv('src/grades.csv')

print("\nИНФОРМАЦИЯ О ДАННЫХ")
print("─" * 50)
print(f"Количество студентов: {len(students)}")
print(f"Количество оценок: {len(grades)}")
print(f"Группы студентов: {', '.join(students['Группа'].unique())}")

# 1. Кто написал контрольную работу, а кто - нет?
print("\n" + "=" * 50)
print("          1. АНАЛИЗ СДАЧИ КОНТРОЛЬНОЙ РАБОТЫ")
print("=" * 50)

submitted_work = students[students['hash'].isin(grades['hash'])].copy()
not_submitted = students[~students['hash'].isin(grades['hash'])].copy()

print(f"Сдали работу: {len(submitted_work)} студентов")
print(f"Не сдали работу: {len(not_submitted)} студентов")

if not not_submitted.empty:
    print(f"\nСтуденты, не сдавшие работу ({len(not_submitted)} чел.):")
    print("─" * 50)

    # Группируем по группам для красивого вывода
    not_submitted_by_group = not_submitted.groupby('Группа')

    for group, group_data in not_submitted_by_group:
        print(f"\nГруппа {group}:")
        for i, (_, student) in enumerate(group_data.iterrows(), 1):
            print(f"   {i:2d}. {student['Фамилия']} {student['Имя']}")
else:
    print("\nВсе студенты сдали работу!")

# Объединяем данные студентов и оценок
merged_data = pd.merge(students, grades, on='hash', how='left')

print(f"\nВсего записей после объединения: {len(merged_data)}")

# 2. Средние оценки по заданиям и группам
print("\n" + "=" * 50)
print("          2. СРЕДНИЕ ОЦЕНКИ ПО ГРУППАМ И ЗАДАНИЯМ")
print("=" * 50)

task_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

# Общая средняя оценка по всем заданиям для каждой группы
merged_data = merged_data.copy()
merged_data['total_score'] = merged_data[task_columns].sum(axis=1)
merged_data['avg_score'] = merged_data[task_columns].mean(axis=1)

# Средняя оценка по каждому заданию в каждой группе
group_task_means = merged_data.groupby('Группа')[task_columns].mean()

print("СРЕДНИЕ ОЦЕНКИ ПО ЗАДАНИЯМ В КАЖДОЙ ГРУППЕ:")
print("─" * 81)

# Создаем красивую таблицу
header = "Группа    " + "".join(f"{f'Зад.{i}':^8}" for i in range(9))
print(header)
print("─" * len(header))

for group in group_task_means.index:
    row = f"{group:<10}"
    for task in task_columns:
        row += f"{group_task_means.loc[group, task]:>7.3f} "
    print(row)

group_overall_means = merged_data.groupby('Группа')['avg_score'].mean()

print("\nОБЩАЯ СРЕДНЯЯ ОЦЕНКА ПО ГРУППАМ:")
print("─" * 50)
for group, score in group_overall_means.items():
    print(f"Группа {group}: {score:.3f}")

# 3. Топ-5 лидеров и отстающих
print("\n" + "=" * 50)
print("          3. ТОП-5 ЛИДЕРОВ И ОТСТАЮЩИХ")
print("=" * 50)

students_with_grades = merged_data.dropna(subset=task_columns).copy()

# Топ-5 лидеров
top_5 = students_with_grades.nlargest(5, 'total_score')[['Группа', 'Фамилия', 'Имя', 'total_score', 'avg_score']]

print("ТОП-5 ЛИДЕРОВ:")
print("─" * 50)
for i, (_, student) in enumerate(top_5.iterrows(), 1):
    print(f"{i}. {student['Группа']} - {student['Фамилия']} {student['Имя']}")
    print(f"   Общий балл: {student['total_score']:.1f} | Средняя оценка: {student['avg_score']:.3f}")

# Топ-5 отстающих
bottom_5 = students_with_grades.nsmallest(5, 'total_score')[['Группа', 'Фамилия', 'Имя', 'total_score', 'avg_score']]

print("\nТОП-5 ОТСТАЮЩИХ:")
print("─" * 50)
for i, (_, student) in enumerate(bottom_5.iterrows(), 1):
    print(f"{i}. {student['Группа']} - {student['Фамилия']} {student['Имя']}")
    print(f"   Общий балл: {student['total_score']:.1f} | Средняя оценка: {student['avg_score']:.3f}")

# 4. Анализ с помощью PCA и кластеризации
print("\n" + "=" * 50)
print("          4. PCA И КЛАСТЕРИЗАЦИЯ")
print("=" * 50)

X = students_with_grades[task_columns].fillna(0)

# Применяем PCA для выделения 2 главных компонент
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
students_with_grades = students_with_grades.copy()
students_with_grades.loc[:, 'PC1'] = principal_components[:, 0]
students_with_grades.loc[:, 'PC2'] = principal_components[:, 1]

print(f"Объясненная дисперсия компонент: {pca.explained_variance_ratio_}")
print(f"Суммарная объясненная дисперсия: {sum(pca.explained_variance_ratio_):.3f}")

# Кластеризация K-means на 4 кластера
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
students_with_grades.loc[:, 'cluster'] = clusters

# Центры кластеров в пространстве PCA
cluster_centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=task_columns)
cluster_centers_pca = pca.transform(cluster_centers_df)

# Визуализация
plt.figure(figsize=(12, 8))

# Точечная диаграмма с цветами по кластерам
scatter = plt.scatter(students_with_grades['PC1'], students_with_grades['PC2'],
                      c=students_with_grades['cluster'], cmap='viridis', alpha=0.7, s=60)

# Отмечаем центры кластеров
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
            c='red', marker='X', s=200, edgecolors='black', linewidth=2)

# Подписываем центры кластеров
for i, center in enumerate(cluster_centers_pca):
    plt.annotate(f'Центр {i}', (center[0], center[1]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=12, fontweight='bold', color='red')

plt.colorbar(scatter, label='Кластер')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)')
plt.title('Кластеризация студентов по оценкам (PCA + K-means)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Анализ кластеров
print("\n" + "=" * 50)
print("          5. АНАЛИЗ КЛАСТЕРОВ")
print("=" * 50)

cluster_stats = students_with_grades.groupby('cluster').agg({
    'total_score': ['mean', 'median'],
    'avg_score': ['mean', 'median'],
    'hash': 'count'
}).round(3)

# Переименовываем индекс и столбцы на русский
cluster_stats.index.name = 'Кластер'
cluster_stats.columns = ['Средний балл', 'Медианный балл', 'Средняя оценка', 'Медианная оценка', 'Количество студентов']

print("СТАТИСТИКА ПО КЛАСТЕРАМ")
print("─" * 98)

# Создаем красивую таблицу
stats_str = cluster_stats.to_string(
    float_format=lambda x: f'{x:.3f}',
    justify='center',
    col_space=12
)
lines = stats_str.split('\n')
print(lines[0])  # заголовок
print('─' * len(lines[0]))
for line in lines[1:]:
    print(line)

# Группы в каждом кластере
print("\nРАСПРЕДЕЛЕНИЕ ГРУПП ПО КЛАСТЕРАМ")
print("─" * 54)

cluster_groups = students_with_grades.groupby(['cluster', 'Группа']).size().unstack(fill_value=0)
cluster_groups.index.name = 'Кластер'

# Добавляем итоговую строку
cluster_groups['ВСЕГО'] = cluster_groups.sum(axis=1)

groups_str = cluster_groups.to_string(
    justify='center',
    col_space=10
)
lines = groups_str.split('\n')
print(lines[0])  # заголовок
print('─' * len(lines[0]))
for line in lines[1:]:
    print(line)

# Детальная информация по каждому кластеру
print("\nДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО КАЖДОМУ КЛАСТЕРУ")
print("─" * 50)

for cluster_num in range(4):
    cluster_data = students_with_grades[students_with_grades['cluster'] == cluster_num]

    # Определяем "уровень" кластера по среднему баллу
    avg_score = cluster_data['total_score'].mean()
    if avg_score >= 4:
        level = "ВЫСОКИЙ"
    elif avg_score >= 2:
        level = "СРЕДНИЙ"
    else:
        level = "НИЗКИЙ"

    print(f"\nКЛАСТЕР {cluster_num} - {level} УРОВЕНЬ")
    print("─" * 50)

    print(f"   Количество студентов: {len(cluster_data)}")

    # Распределение по группам
    group_dist = cluster_data['Группа'].value_counts()
    groups_info = [f"{group} ({count})" for group, count in group_dist.items()]
    print(f"   Группы: {', '.join(groups_info)}")

    print(f"   Средний балл: {cluster_data['total_score'].mean():.2f}")
    print(f"   Медианный балл: {cluster_data['total_score'].median():.2f}")
    print(f"   Средняя оценка: {cluster_data['avg_score'].mean():.3f}")

    # Лучшие студенты
    top_in_cluster = cluster_data.nlargest(3, 'total_score')[['Фамилия', 'Имя', 'Группа', 'total_score', 'avg_score']]
    print(f"   Лучшие студенты:")
    for i, (_, student) in enumerate(top_in_cluster.iterrows(), 1):
        print(f"      {i}. {student['Фамилия']} {student['Имя']} ({student['Группа']})")
        print(f"          Общий балл: {student['total_score']:.1f} | Средняя оценка: {student['avg_score']:.3f}")

# 6. Дополнительная визуализация
print("\n" + "=" * 50)
print("          6. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
print("=" * 50)

plt.figure(figsize=(15, 10))

# График 1: Средние оценки по заданиям по группам
plt.subplot(2, 2, 1)
group_means = merged_data.groupby('Группа')[task_columns].mean().T
group_means.plot(marker='o', ax=plt.gca())
plt.title('Средние оценки по заданиям в каждой группе', fontweight='bold')
plt.xlabel('Номер задания')
plt.ylabel('Средняя оценка')
plt.legend(title='Группа')
plt.grid(True, alpha=0.3)

# График 2: Распределение общих баллов по группам
plt.subplot(2, 2, 2)
group_data = [merged_data[merged_data['Группа'] == group]['total_score'].dropna()
              for group in merged_data['Группа'].unique()]
plt.boxplot(group_data, tick_labels=merged_data['Группа'].unique())
plt.title('Распределение общих баллов по группам', fontweight='bold')
plt.xlabel('Группа')
plt.ylabel('Общий балл')
plt.grid(True, alpha=0.3)

# График 3: Heatmap корреляции между заданиями
plt.subplot(2, 2, 3)
correlation_matrix = merged_data[task_columns].corr()

# Создаем heatmap с помощью imshow
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)

# Добавляем annotations
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8)

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Корреляция между заданиями', fontweight='bold')

# График 4: Кластеры с указанием групп
plt.subplot(2, 2, 4)
groups = students_with_grades['Группа'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))

for i, group in enumerate(groups):
    group_data = students_with_grades[students_with_grades['Группа'] == group]
    plt.scatter(group_data['PC1'], group_data['PC2'],
                color=colors[i], label=group, alpha=0.7, s=60)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)')
plt.title('Кластеры с разбивкой по группам', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Графики визуализации построены успешно!")

# 7. Итоговая статистика
print("\n" + "=" * 50)
print("          7. ИТОГОВАЯ СТАТИСТИКА")
print("=" * 50)

print("ОБЩАЯ СТАТИСТИКА:")
print("─" * 50)
print(f"Общее количество студентов: {len(students)}")
print(f"Сдали работу: {len(submitted_work)} ({len(submitted_work) / len(students) * 100:.1f}%)")
print(f"Не сдали работу: {len(not_submitted)} ({len(not_submitted) / len(students) * 100:.1f}%)")
print(f"Средний балл по всем студентам: {merged_data['total_score'].mean():.2f}")
print(f"Медианный балл по всем студентам: {merged_data['total_score'].median():.2f}")

# Статистика по группам
print("\nСТАТИСТИКА ПО ГРУППАМ:")
print("─" * 84)

group_summary = merged_data.groupby('Группа').agg({
    'hash': 'count',
    'total_score': ['mean', 'median', 'std'],
    'avg_score': 'mean'
}).round(2)

group_summary.columns = ['Кол-во', 'Средний балл', 'Медианный балл', 'Стд. откл.', 'Средняя оценка']

# Красиво форматируем таблицу
summary_str = group_summary.to_string(
    justify='center',
    col_space=12
)
lines = summary_str.split('\n')
print(lines[0])  # заголовок
print('─' * len(lines[0]))
for line in lines[1:]:
    print(line)

# Дополнительные выводы
print("\nКЛЮЧЕВЫЕ ВЫВОДЫ:")
print("─" * 63)

# Находим лучшую и худшую группу
best_group = group_overall_means.idxmax()
worst_group = group_overall_means.idxmin()

print(f"Лучшая группа по успеваемости: {best_group} (средняя оценка: {group_overall_means[best_group]:.3f})")
print(f"Группа, требующая внимания: {worst_group} (средняя оценка: {group_overall_means[worst_group]:.3f})")

# Анализ сдачи работ по группам
submission_stats = merged_data.groupby('Группа').agg(
    Сдали=('total_score', lambda x: x.notna().sum()),
    Всего=('hash', 'count')
).assign(Процент=lambda x: (x['Сдали'] / x['Всего'] * 100).round(1))

best_submission_group = submission_stats['Процент'].idxmax()

print(f"\nЛИДЕР ПО СДАЧЕ РАБОТ: {best_submission_group} " +
      f"({submission_stats.loc[best_submission_group, 'Процент']}% сдали работу)")
