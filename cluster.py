import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from math import pi

csv_file_path = 'new_merged_data.csv'
df = pd.read_csv(csv_file_path)

print("First 5 rows of the dataset:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# Identify columns to exclude (identifiers)
identifier_cols = ['lineId', 'name', 'team', 'season']

# Define the features for clustering by excluding identifiers
features = [col for col in df.columns if col not in identifier_cols]

# Fill missing values if necessary
df[features] = df[features].fillna(df[features].mean())

# Scale the features
scaler = StandardScaler()
df_scaled_features = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(df_scaled_features, columns=features)

# Combine identifier columns with scaled features for later reference
df_preprocessed = pd.concat([df[identifier_cols].reset_index(drop=True), df_scaled], axis=1)
print("\nPreprocessed Data:")
print(df_preprocessed.head())

silhouette_scores = []
ch_scores = []
wcss = []
k_values = range(2, 21) 

for k in k_values:
    #kmeans = KMeans(n_clusters=k, random_state=42)
    
    kmeans = KMeans(n_clusters=k, random_state=20, n_init='auto')
    kmeans.fit(df_scaled)

    cluster_labels = kmeans.fit_predict(df_scaled)
    sil_score = silhouette_score(df_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(df_scaled, cluster_labels)
    wcss_value = kmeans.inertia_ 
    silhouette_scores.append(sil_score)
    ch_scores.append(ch_score)
    wcss.append(wcss_value)
    print(f"For n_clusters = {k}, Silhouette score: {sil_score:.4f}, CH score: {ch_score:.4f}, WCSS: {wcss_value:.4f}")

# Created visulizations for the project presentation later
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.xticks(k_values)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, ch_scores, marker='o', color='orange')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Calinski-Harabasz Score")
plt.title("Calinski-Harabasz Analysis for Optimal k")
plt.xticks(k_values)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', color='green')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.xticks(k_values)
plt.show()


#determined that the optimal number of clusters is 3 and arbitrarily set the random state to 20 for now
kmean = KMeans(n_clusters=3, random_state=20)
kmean.fit(df_scaled)
df_preprocessed['cluster'] = kmean.labels_

# ---we are done with algorithm part---
# Print the first 10 rows of the DataFrame with cluster labels
print("\nClustered Data (first 10 rows):")
print(df_preprocessed[['lineId', 'name', 'team', 'season', 'cluster']].head(10))

# Print the count of rows in each cluster
print("\nCluster Counts:")
print(df_preprocessed['cluster'].value_counts())



# someone figure out how these stats (median and mean for example of each cluster) 
# are calculated / meaning (maddie coe)

pd.set_option('display.max_rows', None)     
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)       
pd.set_option('display.float_format', '{:.6f}'.format) 


cluster_stats = df_preprocessed.drop(['lineId', 'name', 'team', 'season'], axis=1).groupby('cluster').agg(['mean', 'median'])
print(cluster_stats)

# Cluster 0: Good, top-pairing puck movers (high-impact offensive drivers)
# Cluster 2: Middle, solid second-pair roles (balanced)
# Cluster 1: Bad, depth (struggle offensively)


# one person figures out what each clusters represent (calvin)

# someone else figure out how to print out the clusters visually (maddie pyne)


# Scatter Plot of Clusters
# Perform PCA to reduce the dimensionality of the data to 2 components for easier visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_preprocessed.drop(['lineId', 'name', 'team', 'season', 'cluster'], axis=1))
df_preprocessed['pca_1'] = pca_result[:, 0]  # Store the first principal component
df_preprocessed['pca_2'] = pca_result[:, 1]  # Store the second principal component

# Create a scatter plot using the PCA components to visualize clusters
plt.figure(figsize=(10, 6))
custom_palette = {0: "#2ca02c", 1: "#FF0000", 2: "#FFFF00"}  # Custom colors for each cluster
sns.scatterplot(x='pca_1', y='pca_2', hue=df_preprocessed['cluster'], palette=custom_palette, alpha=0.7, data=df_preprocessed)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Clusters')
plt.legend(title='Cluster')
plt.show()


# Defense Boxplots
# Loop through specific defensive features and plot their distributions across clusters
features_to_plot = ['dZoneGiveawaysAgainst', 'mediumDangerShotsAgainst', 'highDangerShotsAgainst', 'blockedShotAttemptsFor', 'hitsFor', 'playStoppedAgainst', 'playContinuedInZoneAgainst', 'playContinuedOutsideZoneAgainst']
for feature in features_to_plot:
    # Create a boxplot for each feature to compare how its distribution differs by cluster
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='cluster', y=feature, data=df_preprocessed, palette=custom_palette)
    plt.title(f'Distribution of {feature} Across Clusters (Defense)')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()


# Offense Boxplots
# Loop through specific offensive features and plot their distributions across clusters
features_to_plot = ['corsiPercentage', 'xGoalsFor', 'shotsOnGoalFor', "flurryScoreVenueAdjustedxGoalsFor", "fenwickPercentage", "playContinuedOutsideZoneFor", 'netGoals']
for feature in features_to_plot:
    # Create a boxplot for each feature to compare how its distribution differs by cluster
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='cluster', y=feature, data=df_preprocessed, palette=custom_palette)
    plt.title(f'Distribution of {feature} Across Clusters (Offense)')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()
    

# For the Radar Charts
numeric_cols = df_preprocessed.drop(['lineId', 'name', 'team', 'season', 'cluster', 'pca_1', 'pca_2'], axis=1)
scaler = MinMaxScaler()
scaler.fit(numeric_cols)  


# Radar Charts for Clusters
def plot_radar_chart(cluster_num, scaler):
    normalized_data = pd.DataFrame(scaler.transform(numeric_cols), columns=numeric_cols.columns)
    normalized_data['cluster'] = df_preprocessed['cluster']
    cluster_means = normalized_data[normalized_data['cluster'] == cluster_num].mean()

    categories = list(cluster_means.index.drop('cluster'))
    values = cluster_means[categories].values.tolist()
    values += values[:1]  

    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.4, label=f'Cluster {cluster_num}')
    ax.plot(angles, values, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, rotation=30)
    ax.set_title(f'Cluster {cluster_num} Profile')
    ax.set_ylim(0, 1)
    plt.legend()
    plt.show()

# Generate radar charts for each cluster
for cluster in df_preprocessed['cluster'].unique():
    plot_radar_chart(cluster, scaler)


# Ranking Score Calculation
# Define the features for offense, defense, and possession
offensive_features = ['xGoalsFor', 'shotsOnGoalFor', 'flurryScoreVenueAdjustedxGoalsFor']
defensive_features = ['blockedShotAttemptsFor', 'hitsFor', 'playStoppedAgainst']
possession_features = ['fenwickPercentage', 'playContinuedOutsideZoneFor', 'netGoals']

# Function to compute a composite ranking score based on weighted averages of offensive, defensive, and possession scores
def compute_ranking_score(row, scaler):
    offensive_score = row[offensive_features].mean()
    defensive_score = row[defensive_features].mean()
    possession_score = row[possession_features].mean()

    ranking_score = (0.4 * offensive_score) + (0.3 * defensive_score) + (0.3 * possession_score)
    return ranking_score

# Scale the features for each category (offense, defense, and possession)
scaler = MinMaxScaler()
df_scaled_offensive = scaler.fit_transform(df[offensive_features])
df_scaled_defensive = scaler.fit_transform(df[defensive_features])
df_scaled_possession = scaler.fit_transform(df[possession_features])

# Add the scaled features back to the DataFrame
df_preprocessed[offensive_features] = df_scaled_offensive
df_preprocessed[defensive_features] = df_scaled_defensive
df_preprocessed[possession_features] = df_scaled_possession

# Apply the ranking score calculation to the DataFrame
df_preprocessed['ranking_score'] = df_preprocessed.apply(lambda row: compute_ranking_score(row, scaler), axis=1)

# Rank players within each cluster based on their computed ranking score
df_preprocessed['rank_within_cluster'] = df_preprocessed.groupby('cluster')['ranking_score'].rank(ascending=False, method='min')

# Rescale the ranking scores to be between 0 and 100
scaler_0_100 = MinMaxScaler(feature_range=(0, 100))
df_preprocessed['ranking_score_scaled'] = scaler_0_100.fit_transform(df_preprocessed[['ranking_score']])

# Sort the DataFrame by cluster and ranking score
sorted_df = df_preprocessed[['lineId', 'name', 'team', 'season', 'cluster', 'ranking_score', 'rank_within_cluster', 'ranking_score_scaled']].sort_values(by=['cluster', 'rank_within_cluster'])

# Save the sorted DataFrame to a CSV file
csv_file_path = 'player_line_ranking.csv'
sorted_df.to_csv(csv_file_path, index=False)


#final step, write a function that fetches rows of dpairs of a particular player and their partner
#gives meaning to our algorithm, coaches use this to see who plays well together and evaluate a certain
#player's performance and player type.
