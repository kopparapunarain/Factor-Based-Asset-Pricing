# Factor-Based Asset Pricing: LSTM + Autoencoder, XGBoost, and SHAP Pipeline

This project builds and benchmarks advanced machine learning pipelines—including LSTM networks, Autoencoders, and XGBoost—to forecast next-month stock returns using a rich set of technical and fundamental indicators. Leveraging a universe of approximately 7,000 equities and 98 predictive features from 2016-07-29 to 2020-01-31, the study aims to identify robust, interpretable signals for equity return prediction and portfolio construction.

---

## Key Findings

- **XGBoost delivers superior predictive accuracy:** XGBoost (Model 3) achieved the best test metrics (MSE ≈ 0.0259, MAE ≈ 0.0881), outperforming both the Autoencoder+LSTM (MSE ≈ 0.0440) and Simple LSTM (MSE ≈ 0.0427) models.
- **Optimal modeling strategies uncovered:** The best results were found using an 8-period look-back window, with the Autoencoder+LSTM model performing best when compressed to 256 latent features.
- **Enhanced explainability with SHAP:** SHAP analysis identified approximately 20 company-specific features (out of 98) as the most influential drivers of next-month returns, providing actionable insights for quantitative portfolio management and risk control.

---
