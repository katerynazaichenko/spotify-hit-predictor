# -----------------------------------------------------------------------------
#                  SCORING AND MACHINE LEARNING PROJECT
# -----------------------------------------------------------------------------

# Overview: 
# This analysis explores whether specific variables in the Spotify 2023 dataset, 
# such as release date, inclusion in Spotify charts, song attributes like 
# valence and accousticness can effectively explain the number of 
# streams for the platform's most popular songs. By leveraging various 
# predictive models, we aim to identify the key drivers behind streaming success 
# and compare model performances. 

# Authors: Jessie Cameron, Gabriela Moravcikova and Kateryna Zaichenko
# Date: 12 January 2025

# -----------------------------------------------------------------------------
#                        INSTALL PACKAGES & SET WD
# -----------------------------------------------------------------------------

# Import libaries 
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_text, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Set working directory
os.chdir(r'C:\Users\jessi\OneDrive\Documents\PSME\FTD\Semester One\Machine Learning')

# -----------------------------------------------------------------------------
#                        IMPORT & CLEAN DATA
# -----------------------------------------------------------------------------

# Load data from the excel file 
df = pd.read_csv('spotify-2023.csv', encoding='latin1', thousands=',')

# Drop row with incorrect streams data
df = df.drop(index=574)

# Check variable type and convert to correct format
print(df.dtypes)

df['streams'] = df['streams'].astype('int64') # convert to integer after dropping row 574
df['in_shazam_charts'] = df['in_shazam_charts'].fillna(0).astype(int) # set nans to 0 

# convert streams into millions
df['streams'] = df['streams'] / 1000000

# Define features and target
features = [
    "artist_count", "released_year", "released_month", "released_day", 
    "in_spotify_playlists", "in_spotify_charts", "in_apple_playlists", 
    "in_apple_charts", "in_deezer_playlists", "in_deezer_charts", "in_shazam_charts", 
    "bpm", "danceability_%", "valence_%", "energy_%", "acousticness_%", 
    "instrumentalness_%", "liveness_%", "speechiness_%"
]
target = "streams"

# Ensure all features and target are in the dataset 
df = df[features + [target]] 

# Handle missing values
df = df.fillna(0)  # Replace missing values with 0

# -----------------------------------------------------------------------------
#                        DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------

# table summary
descriptive_stats = df.describe()
print(descriptive_stats)
descriptive_stats.to_csv('descriptive_statistics.csv', index=True) # export table

# top 10 songs by streams
top_songs = df[['track_name', 'artist(s)_name', 'streams']].sort_values(by='streams', ascending=False).head(10)
print(top_songs)

# count of unique artists
unique_artists_count = df['artist(s)_name'].nunique()
print(f"\nNumber of Unique Artists: {unique_artists_count}")

# average streams by artist
avg_streams_artist = df.groupby('artist(s)_name')['streams'].mean().sort_values(ascending=False)
print(avg_streams_artist.head(10))

#-------------------------------- TOP SONGS ---------------------------------------

# Combine track name and artist name 
top_songs['track_artist'] = top_songs['track_name'] + " - " + top_songs['artist(s)_name']

# Sort by number of streams
top_songs_sorted = top_songs.sort_values(by='streams', ascending=True)

# Plot 
plt.figure(figsize=(12, 8))
plt.barh(top_songs_sorted['track_artist'], top_songs_sorted['streams'], color='#FF00FF')
plt.xlabel('Streams (in Millions)', fontsize=12)
plt.ylabel('Track & Artist', fontsize=12)
plt.title('Top 10 Songs by Spotify Streams', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('top_10_songs.png', dpi=300)
plt.show()

#-------------------------------- CORRELATION  --------------------------------

# Check the correlation
corr = df[features + [target]].select_dtypes(include=['number']).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

#--------------------------- MULTICOLINEARITY  --------------------------------

# Calculate the VIF to detect multicolinearity
vif_data = pd.DataFrame()
vif_data["feature"] = features
vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
vif_data.to_csv('vif_score.csv', index=True) # export table
print(vif_data)

# Drop variables with high VIF ( > 10)
columns_to_drop = ['released_year', 'bpm', 'danceability_%', 'energy_%', 'in_spotify_playlists']
features = [feature for feature in features if feature not in columns_to_drop]

#-------------------------------- NON-LINEARITY  --------------------------------

# Scatter plot of features
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature], y=df['streams'])
    plt.title(f'Scatter Plot of {feature} vs Streams', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Streams (in millions)', fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    # Save the plot with a dynamically adjusted name
    plt.savefig(f'scatter_plot_{feature}.png', dpi=300)
    plt.show()

# Pairplot for all features + target
sns.pairplot(df[features + ['streams']])
plt.tight_layout()
# Save the pairplot
plt.savefig('pairplot_features_vs_streams.png', dpi=300)
plt.show()

# -----------------------------------------------------------------------------
#                       BASELINE - LINEAR REGRESSION
# -----------------------------------------------------------------------------

# Split data into independent (X) and dependent (y) variables
X = df[features]
y = df[target]

# Perform train-test split (20% data testing and 80% training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Print shapes of the datasets for verification
print(f"Training Set Shape: {X_train.shape}")
print(f"Test Set Shape: {X_test.shape}")

# Run regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#----------------------- EVALUATE LINEAR REGRESSION ---------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r_squared)
print("Root Mean Squared Error (RMSE):",rmse)

#-------------------- FEATURE IMPORTANCE ANALYSIS------------------------------ 

coefficients = model.coef_
features = X_train.columns

feature_importance = pd.Series(coefficients, index=features).sort_values(ascending=False)

# Print the feature importance
print("Feature Importance for Linear Regression:")
print(feature_importance)

# Plot the feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh', color='skyblue')
plt.title('Feature Importance for Linear Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# -----------------------------------------------------------------------------
#                             MODEL 1 - LASSO
# -----------------------------------------------------------------------------

# Standardize features as LASSO sensitive to feature scales - prevents bias
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)       # Transform test data

# initalise LASSO model, cross-validation
lasso = LassoCV(cv = 5, random_state = 1) # re-run for fewer and more folds (cv)

# fit the model on the training data - teach relationship between x and y for 80% observations (training)
lasso.fit(X_train_scaled, y_train)

# use trained parameters predict on test data
y_pred = lasso.predict(X_test_scaled)

#----------------------- EVALUATE LASSO MODEL ---------------------------------

# Evaluate LASSO model
mae_lasso = mean_absolute_error(y_test, y_pred)
mse_lasso = mean_squared_error(y_test, y_pred)
rmse_lasso = np.sqrt(mse)
r2_lasso = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae_lasso:.4f}")
print(f"Mean Squared Error (MSE): {mse_lasso:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lasso:.4f}")
print(f"R-squared (R²): {r2_lasso:.4f}")

#------------------------ CHECK RESIDUALS -------------------------------------

# Calculate residuals
residuals_lasso = y_test - y_pred

# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals_lasso, kde=True, bins=25, color='#9370DB')  
plt.title("LASSO Regression - Residual Distribution", fontsize=14)
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("Histogram_lasso.png", dpi=300)
plt.show()

# Residuals vs Predicted Values - 
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals_lasso, alpha=0.5, color='#9370DB', edgecolor='k')
plt.axhline(0, color='black', linestyle='--')
plt.title("LASSO Regression - Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.savefig("Residuals_Predicted_lasso.png", dpi=300)
plt.show()

# --------------------------- CROSS VALIDATION -------------------------------

# Optimal Lambda 
optimal_lambda = lasso.alpha_
print(f"Optimal Lambda (Regularization Strength): {optimal_lambda:.4f}")

# Cross-Validation Results Summary
print("\nAlphas Tested (Lambda Values):", lasso.alphas_)
mean_mse_per_alpha = lasso.mse_path_.mean(axis=1)
print("Mean MSE for Each Lambda:", mean_mse_per_alpha)

# Identify the lambda with minimum MSE
min_mse_index = mean_mse_per_alpha.argmin()
best_alpha = lasso.alphas_[min_mse_index]
print(f"\nAlpha with Minimum MSE: {best_alpha:.4f}")

# Create DataFrame for Cross-Validation Results
cv_results = pd.DataFrame({
    "Lambda (Alpha)": lasso.alphas_,
    "Mean MSE": mean_mse_per_alpha
}).sort_values(by="Mean MSE")

# Display Top 10 Results
print("\nTop 10 Cross-Validation Results:")
print(cv_results.head(10))

# Plot
plt.figure(figsize=(8, 6))

# Main Curve: Mean MSE vs Lambda
plt.plot(
    lasso.alphas_, mean_mse_per_alpha, 
    color='blue', linestyle='-', label='MSE Curve'
)

# Highlight the Optimal Lambda
plt.scatter(
    optimal_lambda, mean_mse_per_alpha.min(), 
    color='red', label='Optimal Lambda', zorder=5
)

# Formatting the Plot
plt.xlabel('Regularization Strength - Lambda', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Cross-Validation MSE for Different Lambda Values', fontsize=14)
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('Cross-Validation_MSE_for_Different_Lambda_Values.png', dpi=300)
plt.show()


#-------------------- FEATURE IMPORTANCE ANALYSIS------------------------------ 

# Extract features dynamically and handle zero coefficients
features = X_train.columns if hasattr(X_train, 'columns') else [f"Feature_{i}" for i in range(X_train_scaled.shape[1])]
lasso_coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": lasso.coef_
})
# Exclude zero coefficients
lasso_coefficients = lasso_coefficients[lasso_coefficients["Coefficient"] != 0].sort_values(by="Coefficient", ascending=False)

print("LASSO Feature Importance:")
print(lasso_coefficients)


# Sort coefficients by absolute value for better visualization
lasso_coefficients['Absolute Coefficient'] = lasso_coefficients['Coefficient'].abs()
lasso_coefficients = lasso_coefficients.sort_values(by="Absolute Coefficient", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(lasso_coefficients['Feature'], lasso_coefficients['Absolute Coefficient'], color='#9370DB')  # Nice purple color
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.title("LASSO Regression - Feature Importance")
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.savefig('feature_importance_lasso.png', dpi=300)
plt.show()

#----------------------------- CHARTS -----------------------------------------

# predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# -----------------------------------------------------------------------------
#                             MODEL 2 - DECISION TREE
# -----------------------------------------------------------------------------

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 10, 15],
    'min_samples_leaf': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',  # Optimize for MSE
    verbose=1
)

# Fit to training data
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# Best model
best_tree = grid_search.best_estimator_

# Fit the Decision Tree based on best model parameters
tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, min_samples_split=2, random_state=42)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

#----------------------- EVALUATE DECISION TREE ---------------------------------

mae_dt = mean_absolute_error(y_test, y_pred)
mse_dt = mean_squared_error(y_test, y_pred)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae_dt)
print("Mean Squared Error (MSE):", mse_dt)
print("Root Mean Squared Error (RMSE):", rmse_dt)
print("R-squared (R²):", r2_dt)



#-------------------- FEATURE IMPORTANCE ANALYSIS------------------------------ 

feature_importance = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=True)

# Print feature importance
print("Feature Importance for Decision Tree:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Decision Tree - Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()  # Adjust layout to avoid cutoff issues
plt.savefig('feature_importance_dt.png', dpi=300)
plt.show()

#-------------------- DECISION TREE VISUALISATION ------------------------------ 

print(export_text(tree, feature_names=X.columns.tolist()))

# Plot visualization
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns.tolist(), filled=True)
plt.show()

# Plot visualization of the tree limited to max depth 3 for simplicity
plt.figure(figsize=(38, 15))
plot_tree(tree, feature_names=X.columns.tolist(), filled=True, max_depth=3,fontsize=15)
plt.show()


#------------------------ CHECK RESIDUALS -------------------------------------

residuals_dt = y_test - y_pred

# Plot residuals histogram
plt.figure(figsize=(8, 6))
sns.histplot(residuals_dt, kde=True, bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Plot residuals vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals_dt, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

#------------------------ CHECK FOR OVERFITTING -------------------------------------

y_train_pred = tree.predict(X_train)

# Calculate R² for training and test sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_pred)

print(f"Training R²: {r2_train}")
print(f"Test R²: {r2_test}")

# ----------------------------- CHARTS -----------------------------------------

# Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree - Actual vs Predicted Values")
plt.tight_layout()
plt.savefig('Actual_vs_Predicted_decision_tree.png', dpi=300)
plt.show()
# ----------------------------------------------------------------------------- 
#                           MODEL 3 - RANDOM FOREST
# ----------------------------------------------------------------------------- 

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,    # Number of trees in the forest
    max_depth=None,      # Allow trees to expand fully unless limited by other parameters
    random_state=1,      # For reproducibility
    n_jobs=-1            # Use all available CPU cores
)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)


#----------------------- EVALUATE RANDOM FOREST MODEL -------------------------

# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print performance metrics
print("\nRandom Forest Model Performance:")
print(f"Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"R-squared (R²): {r2_rf:.4f}")

# ------------------------ CHECK RESIDUALS -------------------------------------

# Calculate residuals
residuals_rf = y_test - y_pred_rf

# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals_rf, kde=True, bins=25, color='#00BFFF')
plt.title("Random Forest - Residual Distribution", fontsize=14)
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("Histogram_rf.png", dpi=300)
plt.show()

# Residuals vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.5, color='#00BFFF', edgecolor='k')
plt.axhline(0, color='black', linestyle='--')
plt.title("Random Forest - Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("Residuals_Predicted_rf.png", dpi=300)
plt.show()

# --------------------------- CROSS VALIDATION ---------------------------------

# Perform cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
cv_rmse_rf = np.sqrt(-cv_scores_rf)

print("\nRandom Forest Cross-Validation Results:")
print(f"Mean CV RMSE: {cv_rmse_rf.mean():.4f}")
print(f"Standard Deviation of CV RMSE: {cv_rmse_rf.std():.4f}")

#-------------------- FEATURE IMPORTANCE ANALYSIS------------------------------ 

# Extract feature importance
feature_importance_rf = rf_model.feature_importances_
importance_df_rf = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance_rf})
importance_df_rf = importance_df_rf.sort_values(by="Importance", ascending=False)

# Print feature importance
print("\nFeature Importance (Random Forest):")
print(importance_df_rf)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df_rf["Feature"], importance_df_rf["Importance"], color='#00BFFF')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Random Forest - Feature Importance", fontsize=16)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Feature_Importance_rf.png', dpi=300)
plt.show()

# ----------------------------- CHARTS -----------------------------------------

# Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest - Actual vs Predicted Values")
plt.tight_layout()
plt.savefig('Actual_vs_Predicted_rf.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------------- 
#                       MODEL 4 - GRADIENT BOOSTING
# ----------------------------------------------------------------------------- 

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(
    n_estimators=300,    # Number of boosting stages (trees)
    learning_rate=0.05,   # Shrinkage rate for each tree
    max_depth=3,         # Depth of each tree
    random_state=1       # For reproducibility
)

# Train the Gradient Boosting model
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# ----------------------- EVALUATE GRADIENT BOOSTING MODEL ---------------------

# Calculate evaluation metrics
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Print performance metrics
print("\nGradient Boosting Model Performance:")
print(f"Mean Absolute Error (MAE): {mae_gb:.4f}")
print(f"Mean Squared Error (MSE): {mse_gb:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gb:.4f}")
print(f"R-squared (R²): {r2_gb:.4f}")

# ------------------------ CHECK RESIDUALS -------------------------------------

# Calculate residuals
residuals_gb = y_test - y_pred_gb

# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals_gb, kde=True, bins=25, color='#FF00FF')
plt.title("Gradient Boosting - Residual Distribution", fontsize=14)
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("Histogram_gb.png", dpi=300)
plt.show()

# Residuals vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_gb, residuals_gb, alpha=0.5, color='#FF00FF', edgecolor='k')
plt.axhline(0, color='black', linestyle='--')
plt.title("Gradient Boosting - Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("Residuals_Predicted_gb.png", dpi=300)
plt.show()

# --------------------------- CROSS VALIDATION ---------------------------------

# Perform cross-validation
cv_scores_gb = cross_val_score(gb_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
cv_rmse_gb = np.sqrt(-cv_scores_gb)

print("\nGradient Boosting Cross-Validation Results:")
print(f"Mean CV RMSE: {cv_rmse_gb.mean():.4f}")
print(f"Standard Deviation of CV RMSE: {cv_rmse_gb.std():.4f}")

# -------------------- FEATURE IMPORTANCE ANALYSIS ----------------------------

# Extract feature importance
feature_importance_gb = gb_model.feature_importances_
importance_df_gb = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance_gb})
importance_df_gb = importance_df_gb.sort_values(by="Importance", ascending=False)

# Print feature importance
print("\nFeature Importance (Gradient Boosting):")
print(importance_df_gb)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df_gb["Feature"], importance_df_gb["Importance"], color='#FF00FF')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Gradient Boosting - Feature Importance", fontsize=16)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Feature_Importance_gb.png', dpi=300)
plt.show()

# ----------------------------- CHARTS -----------------------------------------

# Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Gradient Boosting - Actual vs Predicted Values")
plt.tight_layout()
plt.savefig('Actual_vs_Predicted_gb.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------------- 
#                       MODELS COMPARISION 
# ----------------------------------------------------------------------------- 

scores = {
    "Model": [ "LASSO", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "R-squared (R²)": [ r2_lasso, r2_dt, r2_rf, r2_gb],
    "Mean Squared Error (MSE)": [  mse_lasso,mse_dt, mse_rf, mse_gb],
    "Mean Absolute Error (MAE)": [mae_lasso,  mae_dt,mae_rf, mae_gb],
    "Root Mean Squared Error (RMSE)": [  rmse_lasso,rmse_dt,rmse_rf, rmse_gb],  
}

scores_df = pd.DataFrame(scores)

# Display the table
print(scores_df)

# Plot R-squared scores
plt.figure(figsize=(10, 6))
plt.bar(scores_df["Model"], scores_df["R-squared (R²)"], color=["skyblue", "orange", "green", "red"], edgecolor="black")
plt.title("R-squared Scores Comparison")
plt.ylabel("R-squared (R²)")
plt.xlabel("Models")
plt.ylim(0, 1)  
plt.tight_layout()
plt.show()

# Plot MSE scores
plt.figure(figsize=(10, 6))
plt.bar(scores_df["Model"], scores_df["Mean Squared Error (MSE)"], color=["skyblue", "orange", "green", "red"], edgecolor="black")
plt.title("Mean Squared Error (MSE) Comparison")
plt.ylabel("MSE")
plt.xlabel("Models")
plt.tight_layout()
plt.show()

# Plot MAE scores
plt.figure(figsize=(10, 6))
plt.bar(scores_df["Model"], scores_df["Mean Absolute Error (MAE)"], color=["skyblue", "orange", "green", "red"], edgecolor="black")
plt.title("Mean Absolute Error (MAE) Comparison")
plt.ylabel("MSE")
plt.xlabel("Models")
plt.tight_layout()
plt.show() 

# Plot RMSE scores
plt.figure(figsize=(10, 6))
plt.bar(scores_df["Model"], scores_df["Root Mean Squared Error (RMSE)"], color=["skyblue", "orange", "green", "red"], edgecolor="black")
plt.title("Root Mean Squared Error (RMSE) Comparison")
plt.ylabel("RMSE")
plt.xlabel("Models")
plt.tight_layout()
plt.show()