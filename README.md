# kaggle-march-madness-2025

Predicting NCAA Tournament Outcomes Using a Conservative, Probabilistic Favorite-First Strategy.

This repository contains my full pipeline, experimentation history, and submissions for the Kaggle March Machine Learning Mania 2025 competition.

---

## 🧠 Idea Summary

The key idea behind my approach was to **merge both Men's and Women's tournaments** to significantly increase the number of training samples and stabilize model learning.  
By assuming that both tournaments share a similar structure and competitive distribution, this strategy allowed the model to generalize better with more consistent patterns.

---

## 🔧 Methodology

- **Target**: Predict the point difference between Team1 and Team2.
- **Model**: XGBoost with a custom Cauchy loss function for robustness against outliers.
- **Cross-validation**: Repeated 5-fold CV (10 repeats) to stabilize predictions.
- **Prediction**: Out-of-fold (OOF) predictions, with `num_boost_round` tuned via `xgb.cv`.
- **Post-processing**: Clipped predictions between -30 and 30 to avoid extreme outliers.

### 🛠️ Probability Clipping for Monotonicity

To ensure logical consistency in the predicted probabilities,  
I applied **spline fitting** on the predicted score differences.

However, the spline curve occasionally exhibited **non-monotonic behavior** at the extremes,  
where win probabilities began to **decrease** despite increasing score margins (or vice versa).

To address this, I introduced a **monotonic clipping strategy**:

- If the score difference exceeds a certain **positive threshold**,  
  the probability is clipped to **1.0**.
- If the score difference drops below a certain **negative threshold**,  
  the probability is clipped to **0.0**.

This effectively removes any reversed behavior and ensures:

- Higher score margins always imply higher win probabilities
- Model outputs remain interpretable and trustworthy

🖼️ See the original spline fit below — notice the curve bending back at the ends.
The clipping correction flattens those regions, preserving directional consistency.

![image](https://github.com/user-attachments/assets/05a47743-5d56-4793-a333-56ddcf2bdf79)
### 📉 Impact of Probability Correction

After applying the clipping-based monotonicity fix on the spline-fitted probabilities,  
the Brier score on the 2023 test set improved noticeably:

- **Before Correction**: 0.17957  
- **After Correction**: 0.16836 ✅

This corresponds to a **~6% improvement** in the final evaluation score,  
demonstrating the value of robust post-processing even after model inference.

The correction helped eliminate logical reversals in close matchups  
and stabilized the probability outputs in high-margin scenarios.

## 📊 Notable Techniques

- **Merged Gender Data**: Combined both Men's and Women's tournaments for training. Team IDs were unified and labeled consistently.
- **Regularization**: Used both standard GLM and L1/L2-regularized GLM to estimate team-level quality scores.
- **Feature Design**:
  - Team-specific GLM quality scores
  - Tournament seeds
  - Win percentage & strength ratings
- **Evaluation**: Final model performance measured by MAE using OOF predictions. Hyperparameters were manually tuned.

---

## 🚀 Files

- `2025NCAAmerged_model.ipynb`: Final notebook with merged training, modeling, and submission steps. (notebook/)
- `rawdata/`: (optional) Prepared datasets

---

## 🏁 Results

- Final leaderboard results: _Pending_ (to be updated after full evaluation)

---

## 📝 Future Work

- Train separate models for men and women, and ensemble the predictions
- Use Optuna or Bayesian optimization for hyperparameter tuning
- Incorporate Elo or GLM-based hybrid team rating features

---

## 📬 Contact

For questions, suggestions, or collaboration, feel free to reach out!  
📧 Email: **sungi.hwang.work@gmail.com**
