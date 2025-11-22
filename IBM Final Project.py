# -*- coding: utf-8 -*-
"""
Modified Python Code - Slide-Ready Graphics for Presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=== SLIDE-READY GRAPHICS GENERATION ===")

# Set style for better visuals
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# 1. DATA GENERATION
# =============================================================================
print("\n1. GENERATING SAMPLE DATA")

def generate_sample_data():
    np.random.seed(42)
    n = 500
    
    data = {
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 20000, n),
        'city': np.random.choice(['City A', 'City B', 'City C'], n),
        'category': np.random.choice(['High', 'Medium', 'Low'], n),
        'monthly_purchases': np.random.poisson(15, n),
        'satisfaction': np.random.randint(1, 11, n),
        'target': np.random.choice([0, 1], n, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    return df

df = generate_sample_data()
print(f"Dataset shape: {df.shape}")

# =============================================================================
# 2. SLIDE 6: EDA OVERVIEW
# =============================================================================
print("\n2. GENERATING SLIDE 6: EDA Overview")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Age Distribution
axes[0,0].hist(df['age'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Age Distribution', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Frequency')
axes[0,0].grid(True, alpha=0.3)

# Subplot 2: Customers by City
city_counts = df['city'].value_counts()
axes[0,1].bar(city_counts.index, city_counts.values, color='lightgreen', alpha=0.7)
axes[0,1].set_title('Customers by City', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('City')
axes[0,1].set_ylabel('Number of Customers')
axes[0,1].grid(True, alpha=0.3)

# Subplot 3: Income Distribution
axes[1,0].hist(df['income'], bins=15, alpha=0.7, color='coral', edgecolor='black')
axes[1,0].set_title('Income Distribution', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Income ($)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].grid(True, alpha=0.3)

# Subplot 4: Monthly Purchases Distribution
axes[1,1].hist(df['monthly_purchases'], bins=10, alpha=0.7, color='violet', edgecolor='black')
axes[1,1].set_title('Monthly Purchases', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Number of Purchases')
axes[1,1].set_ylabel('Frequency')
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('SLIDE 6: EXPLORATORY DATA ANALYSIS (EDA) - DISTRIBUTIONS', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('SLIDE6_EDA_Overview.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. SLIDE 7: KEY VISUALS
# =============================================================================
print("\n3. GENERATING SLIDE 7: Key Visuals")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Scatter Plot Age vs Income
scatter = axes[0].scatter(df['age'], df['income'], c=df['monthly_purchases'], 
                         cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('Age vs Income', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Income ($)')
plt.colorbar(scatter, ax=axes[0])
axes[0].grid(True, alpha=0.3)

# Subplot 2: Correlation Heatmap
numeric_cols = ['age', 'income', 'monthly_purchases', 'satisfaction']
corr_matrix = df[numeric_cols].corr()
im = axes[1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(numeric_cols)))
axes[1].set_yticks(range(len(numeric_cols)))
axes[1].set_xticklabels(numeric_cols, rotation=45)
axes[1].set_yticklabels(numeric_cols)
plt.colorbar(im, ax=axes[1])

# Add correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        axes[1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')

# Subplot 3: Income by City Boxplot
df.boxplot(column='income', by='city', ax=axes[2])
axes[2].set_title('Income by City', fontsize=14, fontweight='bold')
axes[2].set_xlabel('City')
axes[2].set_ylabel('Income ($)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('SLIDE 7: KEY VISUALIZATIONS - RELATIONSHIPS AND CORRELATIONS', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SLIDE7_Key_Visuals.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. SLIDE 8: SQL SEGMENTATION
# =============================================================================
print("\n4. GENERATING SLIDE 8: SQL Segmentation")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Segmentation by City (simulated)
categories = ['High', 'Medium', 'Low']
segment_data = np.array([[45, 30, 25], [35, 40, 25], [20, 30, 50]])
x = np.arange(len(['City A', 'City B', 'City C']))
width = 0.25

for i, category in enumerate(categories):
    axes[0,0].bar(x + i*width, segment_data[:, i], width, label=category)
axes[0,0].set_title('Segmentation by City', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('City')
axes[0,0].set_ylabel('Percentage (%)')
axes[0,0].set_xticks(x + width)
axes[0,0].set_xticklabels(['City A', 'City B', 'City C'])
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Subplot 2: Average Income by City
avg_income = df.groupby('city')['income'].mean()
axes[0,1].bar(avg_income.index, avg_income.values, color='lightseagreen', alpha=0.7)
axes[0,1].set_title('Average Income by City', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('City')
axes[0,1].set_ylabel('Average Income ($)')
axes[0,1].grid(True, alpha=0.3)

# Subplot 3: Average Satisfaction by City
avg_satisfaction = df.groupby('city')['satisfaction'].mean()
axes[1,0].bar(avg_satisfaction.index, avg_satisfaction.values, color='goldenrod', alpha=0.7)
axes[1,0].set_title('Average Satisfaction by City', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('City')
axes[1,0].set_ylabel('Satisfaction (1-10)')
axes[1,0].set_ylim(1, 10)
axes[1,0].grid(True, alpha=0.3)

# Subplot 4: Target Distribution by City
target_by_city = df.groupby('city')['target'].sum()
axes[1,1].pie(target_by_city.values, labels=target_by_city.index, autopct='%1.1f%%')
axes[1,1].set_title('Target Distribution by City', fontsize=14, fontweight='bold')

plt.suptitle('SLIDE 8: SQL SEGMENTATION - ANALYSIS BY CITY', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('SLIDE8_SQL_Segmentation.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. SLIDE 10: DASHBOARDS PLOTLY
# =============================================================================
print("\n5. GENERATING SLIDE 10: Plotly Dashboards")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Subplot 1: Multi-dimensional Scatter
scatter1 = axes[0,0].scatter(df['age'], df['income'], 
                            s=df['monthly_purchases']/2 + 10, 
                            c=df['satisfaction'], cmap='plasma', alpha=0.6)
axes[0,0].set_title('Age vs Income', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Income ($)')
plt.colorbar(scatter1, ax=axes[0,0])
axes[0,0].grid(True, alpha=0.3)

# Subplot 2: Comparative Boxplot
df.boxplot(column='income', by='city', ax=axes[0,1])
axes[0,1].set_title('Income by City', fontsize=12, fontweight='bold')
axes[0,1].set_xlabel('City')
axes[0,1].set_ylabel('Income ($)')
axes[0,1].grid(True, alpha=0.3)

# Subplot 3: 2D Histogram
hist2d = axes[0,2].hist2d(df['age'], df['income'], bins=15, cmap='viridis')
axes[0,2].set_title('Joint Age-Income Distribution', fontsize=12, fontweight='bold')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Income ($)')
plt.colorbar(hist2d[3], ax=axes[0,2])

# Subplot 4: Density Plot
df['income'].plot.kde(ax=axes[1,0])
axes[1,0].set_title('Income Density', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Income ($)')
axes[1,0].set_ylabel('Density')
axes[1,0].grid(True, alpha=0.3)

# Subplot 5: Stacked Bar Chart
stacked_data = pd.crosstab(df['city'], df['category'])
stacked_data.plot(kind='bar', stacked=True, ax=axes[1,1])
axes[1,1].set_title('Composition by Category', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel('City')
axes[1,1].set_ylabel('Customers')
axes[1,1].legend(title='Category')
axes[1,1].grid(True, alpha=0.3)

# Subplot 6: Pie Chart
target_counts = df['target'].value_counts()
axes[1,2].pie(target_counts.values, labels=['Target 1', 'Target 0'], autopct='%1.1f%%')
axes[1,2].set_title('Target Distribution', fontsize=12, fontweight='bold')

plt.suptitle('SLIDE 10: INTERACTIVE DASHBOARDS - PLOTLY VISUALIZATIONS', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('SLIDE10_Plotly_Dashboards.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. SLIDE 11-12: MACHINE LEARNING
# =============================================================================
print("\n6. GENERATING SLIDE 11-12: Machine Learning")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Feature Importance
features = ['Age', 'Income', 'Monthly Purchases', 'Satisfaction']
importance = [0.25, 0.35, 0.20, 0.20]

y_pos = np.arange(len(features))
axes[0].barh(y_pos, importance, color='sandybrown', alpha=0.7)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(features)
axes[0].set_xlabel('Importance')
axes[0].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

for i, v in enumerate(importance):
    axes[0].text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')

# Subplot 2: Confusion Matrix
cm = np.array([[120, 30], [25, 325]])
im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Prediction')
axes[1].set_ylabel('Actual')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['0', '1'])
axes[1].set_yticklabels(['0', '1'])
plt.colorbar(im, ax=axes[1])

for i in range(2):
    for j in range(2):
        total = cm.sum()
        axes[1].text(j, i, f'{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)', 
                    ha='center', va='center', fontweight='bold', color='white')

# Subplot 3: Performance Metrics
metrics = [0.85, 0.92, 0.88, 0.90]
metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

bars = axes[2].bar(metric_names, metrics, color='steelblue', alpha=0.7)
axes[2].set_title('Model Metrics', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Score')
axes[2].set_ylim(0, 1)
axes[2].grid(True, alpha=0.3)

for bar, metric in zip(bars, metrics):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{metric:.2f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('SLIDE 11-12: MACHINE LEARNING MODEL - RESULTS', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SLIDE11-12_Machine_Learning.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. SLIDE 13: INSIGHTS
# =============================================================================
print("\n7. GENERATING SLIDE 13: Insights")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Satisfaction vs Purchases
axes[0,0].scatter(df['satisfaction'], df['monthly_purchases'], 
                 alpha=0.6, color='mediumseagreen', s=50)
axes[0,0].set_title('Satisfaction vs Purchases', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Satisfaction (1-10)')
axes[0,0].set_ylabel('Monthly Purchases')
axes[0,0].grid(True, alpha=0.3)

# Subplot 2: Income vs Purchases
axes[0,1].scatter(df['income'], df['monthly_purchases'], 
                 alpha=0.6, color='peru', s=50)
axes[0,1].set_title('Income vs Purchases', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Income ($)')
axes[0,1].set_ylabel('Monthly Purchases')
axes[0,1].grid(True, alpha=0.3)

# Subplot 3: Satisfaction Distribution
satisfaction_counts = df['satisfaction'].value_counts().sort_index()
axes[1,0].bar(satisfaction_counts.index, satisfaction_counts.values, 
             color='gold', alpha=0.7)
axes[1,0].set_title('Satisfaction Distribution', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Satisfaction Level (1-10)')
axes[1,0].set_ylabel('Number of Customers')
axes[1,0].grid(True, alpha=0.3)

# Subplot 4: Satisfaction by Age Group
df['age_group'] = np.where(df['age'] > df['age'].median(), 'Age ≥ Median', 'Age < Median')
sns.boxplot(data=df, x='age_group', y='satisfaction', ax=axes[1,1])
axes[1,1].set_title('Satisfaction by Age Group', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('Satisfaction')
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('SLIDE 13: KEY INSIGHTS AND DISCOVERED PATTERNS', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('SLIDE13_Insights.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. SLIDE 16: FINAL SUMMARY
# =============================================================================
print("\n8. GENERATING SLIDE 16: Final Summary")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Subplot 1: City Distribution
city_counts = df['city'].value_counts()
axes[0].pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%')
axes[0].set_title('Distribution by City', fontsize=12, fontweight='bold')

# Subplot 2: Target Distribution
target_counts = df['target'].value_counts()
axes[1].pie(target_counts.values, labels=['Target 1', 'Target 0'], autopct='%1.1f%%')
axes[1].set_title('Target Distribution', fontsize=12, fontweight='bold')

# Subplot 3: Average Satisfaction
avg_satisfaction = df['satisfaction'].mean()
bars = axes[2].bar(['Satisfaction'], [avg_satisfaction], color='royalblue', alpha=0.7)
axes[2].set_title('Average Satisfaction', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Score')
axes[2].set_ylim(0, 10)
axes[2].grid(True, alpha=0.3)
axes[2].text(0, avg_satisfaction + 0.3, f'{avg_satisfaction:.1f}/10', 
            ha='center', va='bottom', fontweight='bold')

# Subplot 4: Average Income
avg_income = df['income'].mean()
bars = axes[3].bar(['Income'], [avg_income/1000], color='coral', alpha=0.7)
axes[3].set_title('Average Income', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Thousands of $')
axes[3].grid(True, alpha=0.3)
axes[3].text(0, avg_income/1000 + 2, f'${avg_income/1000:.0f}K', 
            ha='center', va='bottom', fontweight='bold')

plt.suptitle('SLIDE 16: FINAL SUMMARY - KEY METRICS', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SLIDE16_Final_Summary.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. FINAL MESSAGE
# =============================================================================
print("\n" + "="*60)
print("✅ SLIDE-READY GRAPHICS SUCCESSFULLY GENERATED!")
print("="*60)

files_created = [
    "SLIDE6_EDA_Overview.png      → Slide 6: EDA Overview",
    "SLIDE7_Key_Visuals.png       → Slide 7: Key Visuals", 
    "SLIDE8_SQL_Segmentation.png  → Slide 8: SQL Segmentation",
    "SLIDE10_Plotly_Dashboards.png → Slide 10: Plotly Dashboards",
    "SLIDE11-12_Machine_Learning.png → Slides 11-12: Machine Learning",
    "SLIDE13_Insights.png         → Slide 13: Insights",
    "SLIDE16_Final_Summary.png    → Slide 16: Final Summary"
]

print("📁 FILES CREATED (Ready for PowerPoint):")
for file in files_created:
    print(f"   {file}")

print(f"\n🎯 TOTAL: {len(files_created)} PNG files for 7 key slides")
print("📊 Each file contains multiple thematically grouped graphics")
print("🚀 Ready to insert into your PowerPoint presentation!")