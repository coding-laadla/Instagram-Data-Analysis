import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
df = pd.read_csv("/content/instagram_analysis_700.csv")
print("Shape:", df.shape)
df.info()
df.describe()

# Convert upload_date to datetime
df['upload_date'] = pd.to_datetime(df['upload_date'])

# Extract time-based features
df['day'] = df['upload_date'].dt.day_name()
df['month'] = df['upload_date'].dt.month
df['hour'] = df['upload_date'].dt.hour  # simulated hour (assumed)

# Engagement = Likes + Comments
df['engagement'] = df['likes'] + df['comments']

# Engagement rate proxy (per follower growth)
df['engagement_per_follow'] = df['engagement'] / (df['follows_gained'] + 1)

df[['likes', 'comments', 'follows_gained', 'engagement', 'engagement_per_follow']].head()

day_engagement = df.groupby('day')['engagement'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
day_engagement.plot(kind='bar')
plt.title("Average Engagement by Day")
plt.ylabel("Avg Engagement")
plt.show()

monthly_engagement = df.groupby('month')['engagement'].mean()

plt.figure(figsize=(10,5))
monthly_engagement.plot(marker='o')
plt.title("Average Engagement by Month")
plt.xlabel("Month")
plt.ylabel("Avg Engagement")
plt.show()

tag_performance = df.groupby('tag').agg({
    'likes':'mean',
    'comments':'mean',
    'follows_gained':'mean',
    'engagement':'mean'
}).sort_values(by='engagement', ascending=False)

tag_performance

plt.figure(figsize=(12,6))
sns.barplot(
    x=tag_performance.index,
    y=tag_performance['engagement']
)
plt.xticks(rotation=45)
plt.title("Average Engagement by Hashtag")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='engagement',
    y='follows_gained',
    hue='tag',
    alpha=0.6
)
plt.title("Engagement vs Follower Growth")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(
    df[['likes','comments','follows_gained','engagement']].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()

print("Top 5 Hashtags by Engagement:")
print(tag_performance.head())
