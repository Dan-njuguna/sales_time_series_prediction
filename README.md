# Time Series Analysis Experiments 😊

Welcome to a collection of experiments focused on forecasting retail sales using time series analysis. With historical retail data spanning the last 4 years, this project aims to predict future sales and uncover trends over time.

## Project Overview

- **Data Source:** The dataset is sourced from [Kaggle Datasets](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) and contains detailed sales records across various U.S. states.
- **Dataset Details:** The data comprises 9,800 rows and 18 columns, offering a comprehensive view of the retail store's operations.
- **Objective:**
  - Predict product names and sales values based on customer information and specified future date ranges.
  - Develop a reusable codebase that supports large-scale prediction, retraining, data versioning, and model versioning.

## Dataset Overview

The work is conducted on a prominent dataset that features retailer sales data in the US.
The dataset exhibits sales records over 4 years: between January 2015 and December 2018. It contains 18 columns (features) and 9,800 rows (instances).
The columns are described as follows:

- Row ID: An integer value representing a unique sequence ID for the rows in the dataset.
- Order ID: An alphanumeric identifier for the purchase order. It can appear multiple times if the order included multiple items.
- Order Date: The date when the purchase order was placed.
- Ship Date: The date when the order was shipped out from the retailer.
- Ship Mode: A string indicating the priority level of the transportation (e.g., First Class, Standard Class).
- Customer ID: An alphanumeric customer identifier that appears to be anonymized.
- Customer Name: A string showing the name of the customer, which also appears to be anonymized.
- Segment: A string describing the kind of customer: Consumer, Corporate, or Home Office.
- Country: A string indicating that the dataset only includes orders from the United States.
- City: A string representing the US city from which the order originated.
- State: A string representing the US state from which the order originated.
- Post Code: An alphanumeric value that appears to be the zip code.
- Region: A string indicating a high-level geographical division of the US (e.g., East, West).
- Product ID: An alphanumeric product identifier.
- Category: A string describing the type of product: Furniture, Office Supplies, or Technology.
- Sub-Category: A string for the sub-type of the product (e.g., Chairs, Phones, Envelopes).
- Product Name: A string showing the name of the item sold.
- Sales: A float representing the monetary amount of the sold products (assumed to be in USD).

## Approach

- **Experimentation:**Multiple forecasting methods will be employed to achieve optimal predictions.
- **Reusability:**
  The design emphasizes code reusability to facilitate future enhancements and scalability.

Thank you for exploring this project. Let's dive into time series forecasting! 🚀
