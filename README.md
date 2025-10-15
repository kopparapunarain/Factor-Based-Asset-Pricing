LSTM + Autoencoder, XGBoost and SHAP pipeline to predict next-month stock returns from technical & fundamental indicators (2016–2020).

README blurb (short paragraph, 2–3 sentences)
This project trains and compares time-series and tree-based models to forecast next-month stock returns using a universe of ~7k equities (2016-07-29 to 2020-01-31) and 98 predictors. Implementations include an Autoencoder+LSTM, a simple LSTM, and an XGBoost regressor, with SHAP used for model interpretability to surface the most influential company features. Results show XGBoost achieves the lowest test MSE, while LSTM variants provide useful sequence representations for feature engineering. 

B213801

Key findings (3 bullets)

XGBoost outperforms neural models: XGBoost (Model 3) produced the best test metrics (MSE ≈ 0.0259, MAE ≈ 0.0881) versus Autoencoder+LSTM (MSE ≈ 0.0440) and Simple LSTM (MSE ≈ 0.0427). 

B213801

Optimal hyperparameters found: Best look-back (window) length is 8 time steps and the Autoencoder + LSTM performed best with 256 latent features during experiments. 

B213801

Interpretable drivers via SHAP: SHAP analysis highlights ~20 company features (out of 98) that most strongly influence next-month returns, providing actionable signals for portfolio construction and risk filtering.
