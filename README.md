# Superstore Sales & Profit Analytics Dashboard

## Overview

This project looks at sales and profit data and turns it into an interactive dashboard using Python and Streamlit. The goal was to explore how discounts, categories, and customer segments affect profit, and to build something that lets a user explore those patterns visually.

This project covers data cleaning, analysis, modelling, and then presenting the results in a simple dashboard.

---

# Dataset

This project uses the Superstore dataset from Kaggle:

[https://www.kaggle.com/datasets/vivek468/superstore-dataset-final](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

It contains sales data like sales amount, profit, discount, category, segment, and dates. The data is anonymised and doesn’t include personal information.

---

# Main Question

Discounts help drive sales, but they can also reduce profit. The main thing I wanted to test was whether higher discounts are generally linked to lower profit.

Along the way I also explored:

Which product categories generate the most sales vs the most profit

Which customer segments are most profitable

How profit distribution differs across categories

Whether shipping time has any relationship with profit

How profit changes over time (monthly trends)

Whether profit can be reasonably predicted from order features using a regression model

---

# Project Files

```
Dataset Cleaning and EDA.ipynb
Modelling and Regression.ipynb
streamlit_app.py
```

* Cleaning script — prepares and fixes the dataset
* Modelling script — runs analysis and regression model
* Streamlit app — interactive dashboard

---

# Data Preparation

The cleaning script loads the raw dataset and prepares it for analysis.

Main steps:

* Fix column types
* Convert date columns
* Check for missing values
* Create extra time fields (year and month)
* Save a cleaned version for reuse

The cleaned file is what the modelling and dashboard use.

---

# Exploration

I started with basic exploration to understand the data:

* Summary statistics
* Sales and profit distributions
* Comparisons by category and segment
* Scatter plots of discount vs profit

This helped confirm that heavy discounts often line up with lower profit, with some exceptions.

---

# Modelling

I built a regression model to predict profit using order features.

Inputs include:

* Sales
* Discount
* Quantity
* Category
* Segment
* Region
* Shipping time (calculated from dates)

Steps:

* Scale numeric values
* Encode categories
* Train a linear regression model
* Check MAE, RMSE, and R²

I also looked at the model coefficients to see which features push predictions up or down.

---

# Dashboard

The dashboard is built with Streamlit.

Run it with:

```
streamlit run "streamlit app v4.py"
```

## Controls

Users can filter by:

* Region
* Category
* Segment

## Metrics shown

* Total sales
* Total profit
* Average discount
* Number of orders

## Charts included

* Sales by category (bar)
* Profit by segment (bar)
* Discount vs profit (scatter)
* Profit by category (box plot)
* Monthly sales trend (line)
* Discount distribution (histogram)

The idea is that each chart answers a simple business question.

---

# AI Tools

I used AI tools like ChatGPT and GitHub's AI pilot at points to help with makng unified markdown, project suggestions and debugging, but all code was reviewed and tested before being kept. I treated suggestions like hints, not final answers.

---

# Notes and Limits

* Linear regression is simple and may miss more complex patterns
* Results depend on this specific dataset
* The dashboard summary text is rule-based, not generated live
* This should be used for exploration, not automatic decisions

---

# Possible Next Steps

If I extended this further, I would:

* Try nonlinear models
* Add automated retraining
* Add anomaly detection for extreme discounts
* Connect to a live data source instead of static files
