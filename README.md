### Introduction

### Customer Segmentation in SE using K-Means Clustering and Feature Engineering
This project purpose to divide customers into segments based on K-Means Clustering, which represent best marketing strategies according to the customer sales behavior, characteristics, and needs.

### Usage

Clone the repository:
```bash
git clone https://github.com/CVliner/se_customer_segmentation_k_means.git
```
```bash
cd se_customer_segmentation_k_means
```
Place the Online Retail.xlsx dataset in the project directory and run the script:
```bash
python se_customer_segmentation_k_means.py
```

### **Dataset:**

The dataset obtained from database of SE retail, contains all the transactions occurring between 01/12/2010 and 09/12/2018 for Central Asia customers. The attributes include invoice number, product code, product name, quantity per transaction, invoice date, product unit price, customerID, and country. The company mainly sells electrical systems, automation and switches. Many customers of the company are distrubitors, system integrators and design companies.


### **Results:**

<p align="center">

<img src="https://github.com/CVliner/se_customer_segmentation_k_means/blob/main/pics/Segmentation_K_Means.png" alt="cluster" width="600" height="600">

- Cluster 2 represents the high-value customers, characterized by the highest number of orders, frequency, and the most recent transactions.
- Cluster 1 consists of lost customers who rarely place orders and generate the lowest sales.
- Cluster 0 includes at-risk and loyal customers, who exhibit medium values in terms of frequency, recency, and monetary metrics.
- Across all clusters, high monetary value is associated with a high frequency of orders and more recent transactions.
 
### **Recommendations:**

The company can design targeted marketing campaigns for various customer segments to boost revenue. To achieve this, the company might provide incentives to low-value customers to maintain their engagement and encourage more frequent purchases. Conversely, high-value customers could receive special benefits, such as exclusive discounts and early access to new products. As conclusion, the approach will depend on the company's specific business objectives.
 


