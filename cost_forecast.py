


from google.cloud import bigquery

# Instantiate a BigQuery client
client = bigquery.Client()

# Define your SQL query
sql = """
CREATE OR REPLACE MODEL `project.dataset.billing_forecast_model`
OPTIONS(model_type='ARIMA', time_series_timestamp_col='invoice_month', time_series_data_col='cost') AS
SELECT
  invoice_month,
  SUM(cost) as cost
FROM `project.dataset.billing_data`
GROUP BY invoice_month
ORDER BY invoice_month ASC
"""

# Run the query
job = client.query(sql)
job.result()  # Wait for the job to finish