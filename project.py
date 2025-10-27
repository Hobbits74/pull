import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
print("")
print("CROCODILE SPECIES DATASET ANALYSIS")
# Загрузка данных
df = pd.read_csv('crocodile_dataset.csv')

print("")
print("[БАЗОВАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ]")
print(f"Количество наблюдений: {len(df)}")
print(f"Количество признаков: {len(df.columns)}")
print("")
print("Колонки датасета:")
print(df.columns.tolist())

# Информация о данных
print("")
print("СТРУКТУРА ДАННЫХ")
df.info()

# Статистика
print("")
print("СТАТИСТИЧЕСКОЕ ОПИСАНИЕ")
print(df.describe())

# Проверка пропущенных значений
print("")
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Пропущено': missing,
    'Процент': missing_percent
})
print(missing_df[missing_df['Пропущено'] > 0])

#2. EXPLORATORY DATA ANALYSIS 

# 2.1 Распределение видов
print("")
print("АНАЛИЗ ВИДОВ")


species_count = df['Common Name'].value_counts()
print("")
print("Топ-10 наиболее наблюдаемых видов:")
print(species_count.head(10))

plt.figure(figsize=(14, 6))
species_count.head(15).plot(kind='barh')
plt.title('Топ-15 видов по количеству наблюдений', fontsize=14, fontweight='bold')
plt.xlabel('Количество наблюдений')
plt.ylabel('Вид')
plt.tight_layout()
plt.savefig('species_distribution.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: species_distribution.png")
plt.close()

# 2.2 Распределение по семействам
print("")
print("АНАЛИЗ СЕМЕЙСТВ")


family_count = df['Family'].value_counts()
print("")
print("Распределение по семействам:")
print(family_count)

# 2.3 Анализ размеров (длина и вес)
print("")
print("АНАЛИЗ РАЗМЕРОВ")
print("")
print("Длина (м):")
print(f"  Минимум: {df['Observed Length (m)'].min():.2f}")
print(f"  Максимум: {df['Observed Length (m)'].max():.2f}")
print(f"  Среднее: {df['Observed Length (m)'].mean():.2f}")
print(f"  Медиана: {df['Observed Length (m)'].median():.2f}")

print("")
print("Вес (кг):")
print(f"  Минимум: {df['Observed Weight (kg)'].min():.2f}")
print(f"  Максимум: {df['Observed Weight (kg)'].max():.2f}")
print(f"  Среднее: {df['Observed Weight (kg)'].mean():.2f}")
print(f"  Медиана: {df['Observed Weight (kg)'].median():.2f}")

# Графики распределения размеров
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Гистограмма длины
axes[0, 0].hist(df['Observed Length (m)'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Распределение длины', fontweight='bold')
axes[0, 0].set_xlabel('Длина (м)')
axes[0, 0].set_ylabel('Частота')

# Гистограмма веса
axes[0, 1].hist(df['Observed Weight (kg)'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Распределение веса', fontweight='bold')
axes[0, 1].set_xlabel('Вес (кг)')
axes[0, 1].set_ylabel('Частота')

# Boxplot длины
axes[1, 0].boxplot(df['Observed Length (m)'].dropna(), vert=False)
axes[1, 0].set_title('Boxplot длины', fontweight='bold')
axes[1, 0].set_xlabel('Длина (м)')

# Boxplot веса
axes[1, 1].boxplot(df['Observed Weight (kg)'].dropna(), vert=False)
axes[1, 1].set_title('Boxplot веса', fontweight='bold')
axes[1, 1].set_xlabel('Вес (кг)')

plt.tight_layout()
plt.savefig('size_distributions.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: size_distributions.png")
plt.close()

# 2.4 Корреляция длины и веса
plt.figure(figsize=(10, 6))
plt.scatter(df['Observed Length (m)'], df['Observed Weight (kg)'], alpha=0.5)
plt.title('Зависимость веса от длины', fontsize=14, fontweight='bold')
plt.xlabel('Длина (м)')
plt.ylabel('Вес (кг)')
plt.grid(True, alpha=0.3)

# Добавление линии тренда
z = np.polyfit(df['Observed Length (m)'].dropna(), 
               df['Observed Weight (kg)'].dropna(), 2)
p = np.poly1d(z)
x_line = np.linspace(df['Observed Length (m)'].min(), df['Observed Length (m)'].max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label='Тренд')
plt.legend()

plt.tight_layout()
plt.savefig('length_weight_correlation.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: length_weight_correlation.png")
plt.close()

# Корреляция
correlation = df[['Observed Length (m)', 'Observed Weight (kg)']].corr()
print("")
print(f"Корреляция длины и веса: {correlation.iloc[0, 1]:.3f}")

# 2.5 Анализ по возрастным группам
print("")
print("АНАЛИЗ ПО ВОЗРАСТНЫМ ГРУППАМ")


age_count = df['Age Class'].value_counts()
print("")
print("Распределение по возрастным группам:")
print(age_count)

# Средние размеры по возрастным группам
age_stats = df.groupby('Age Class')[['Observed Length (m)', 'Observed Weight (kg)']].agg(['mean', 'std'])
print("")
print("Средние размеры по возрастным группам:")
print(age_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Подготовка данных для боксплотов
age_order = ['Hatchling', 'Juvenile', 'Subadult', 'Adult']
colors_age = ['#FFE5B4', '#FFB347', '#FF8C42', '#FF6B35']

# Boxplot длины по возрасту
bp1 = axes[0].boxplot([df[df['Age Class'] == age]['Observed Length (m)'].dropna() 
                        for age in age_order],
                       labels=age_order,
                       patch_artist=True,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
for patch, color in zip(bp1['boxes'], colors_age):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(2)
axes[0].set_title('Length by Age Groups', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age Group', fontsize=12)
axes[0].set_ylabel('Length (m)', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Boxplot веса по возрасту
bp2 = axes[1].boxplot([df[df['Age Class'] == age]['Observed Weight (kg)'].dropna() 
                        for age in age_order],
                       labels=age_order,
                       patch_artist=True,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
for patch, color in zip(bp2['boxes'], colors_age):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(2)
axes[1].set_title('Weight by Age Groups', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age Group', fontsize=12)
axes[1].set_ylabel('Weight (kg)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: age_analysis.png")
plt.close()

# 2.6 Анализ по полу
print("")
print("АНАЛИЗ ПО ПОЛУ")


sex_count = df['Sex'].value_counts()
print("")
print("Распределение по полу:")
print(sex_count)

sex_stats = df.groupby('Sex')[['Observed Length (m)', 'Observed Weight (kg)']].agg(['mean', 'std'])
print("")
print("Средние размеры по полу:")
print(sex_stats)

# 2.7 Географический анализ
print("")
print("ГЕОГРАФИЧЕСКИЙ АНАЛИЗ")

country_count = df['Country/Region'].value_counts()
print("")
print("Топ-15 стран/регионов по количеству наблюдений:")
print(country_count.head(15))

plt.figure(figsize=(14, 6))
country_count.head(15).plot(kind='barh')
plt.title('Топ-15 стран/регионов по наблюдениям', fontsize=14, fontweight='bold')
plt.xlabel('Количество наблюдений')
plt.ylabel('Страна/Регион')
plt.tight_layout()
plt.savefig('geographic_distribution.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: geographic_distribution.png")
plt.close()

# 2.8 Анализ типов среды обитания
print("")
print("АНАЛИЗ СРЕДЫ ОБИТАНИЯ")

habitat_count = df['Habitat Type'].value_counts()
print("")
print("Распределение по типам среды обитания:")
print(habitat_count)

plt.figure(figsize=(10, 6))
habitat_count.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Распределение по типам среды обитания', fontsize=14, fontweight='bold')
plt.xlabel('Тип среды обитания')
plt.ylabel('Количество наблюдений')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('habitat_distribution.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: habitat_distribution.png")
plt.close()

# 2.9 Анализ статуса охраны
print("")
print("АНАЛИЗ СТАТУСА ОХРАНЫ")
conservation_count = df['Conservation Status'].value_counts()
print("")
print("Распределение по статусу охраны:")
print(conservation_count)

plt.figure(figsize=(10, 6))
colors_conservation = ['green', 'yellow', 'orange', 'red']
conservation_count.plot(kind='bar', color=colors_conservation, edgecolor='black')
plt.title('Распределение по статусу охраны (IUCN)', fontsize=14, fontweight='bold')
plt.xlabel('Статус охраны')
plt.ylabel('Количество наблюдений')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('conservation_status.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: conservation_status.png")
plt.close()

# 2.10 Временной анализ
print("")
print("ВРЕМЕННОЙ АНАЛИЗ")


# Преобразование даты
df['Date of Observation'] = pd.to_datetime(df['Date of Observation'], format='%d-%m-%Y', errors='coerce')
df['Year'] = df['Date of Observation'].dt.year
df['Month'] = df['Date of Observation'].dt.month

yearly_count = df['Year'].value_counts().sort_index()
print("")
print("Наблюдения по годам:")
print(yearly_count)

plt.figure(figsize=(12, 6))
yearly_count.plot(kind='line', marker='o', linewidth=2, markersize=6)
plt.title('Динамика наблюдений по годам', fontsize=14, fontweight='bold')
plt.xlabel('Год')
plt.ylabel('Количество наблюдений')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
print("")
print("[OK] График сохранён: temporal_analysis.png")
plt.close()

#3. СВОДНЫЙ ОТЧЁТ 
print("")
print("[ОСНОВНЫЕ ВЫВОДЫ]")
print("")
print("1. ВИДОВОЕ РАЗНООБРАЗИЕ:")
print(f"   - Всего уникальных видов: {df['Common Name'].nunique()}")
print(f"   - Семейств: {df['Family'].nunique()}")
print(f"   - Родов: {df['Genus'].nunique()}")
print("")
print("2. РАЗМЕРНЫЕ ХАРАКТЕРИСТИКИ:")
print(f"   - Средняя длина: {df['Observed Length (m)'].mean():.2f} м")
print(f"   - Средний вес: {df['Observed Weight (kg)'].mean():.2f} кг")
print(f"   - Корреляция длина-вес: {correlation.iloc[0, 1]:.3f}")
print("")
print("3. ГЕОГРАФИЯ:")
print(f"   - Наблюдения в {df['Country/Region'].nunique()} странах/регионах")
print(f"   - Типов среды обитания: {df['Habitat Type'].nunique()}")
print("")
print("4. ОХРАННЫЙ СТАТУС:")
print(f"   - Endangered/Critically Endangered: {len(df[df['Conservation Status'].isin(['Endangered', 'Critically Endangered'])])} наблюдений")
print(f"   - Процент угрожаемых: {len(df[df['Conservation Status'].isin(['Endangered', 'Critically Endangered'])]) / len(df) * 100:.1f}%")
print("")
print("5. ВРЕМЕННОЙ ПЕРИОД:")
print(f"   - С {df['Year'].min():.0f} по {df['Year'].max():.0f} год")
print(f"   - Период: {df['Year'].max() - df['Year'].min():.0f} лет")

print("")
print("")
print("Сохранённые графики:")
print("  1. species_distribution.png")
print("  2. size_distributions.png")
print("  3. length_weight_correlation.png")
print("  4. age_analysis.png")
print("  5. geographic_distribution.png")
print("  6. habitat_distribution.png")
print("  7. conservation_status.png")
print("  8. temporal_analysis.png")
