from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import cast, col, rand, round

# Initialize SparkSession
spark = SparkSession.builder.appName("SimpleLinearRegression").getOrCreate()

# Generate sample data
num_samples = 100
slope = 2.0
intercept = 5.0
noise_std = 1.0

data = (
    spark.range(0, num_samples)
    .toDF("id")
    .withColumn("x", round(rand() * 10, 2))
    .withColumn("x_double", col("x").cast("double"))  # Cast x to double
    .withColumn("y", round(slope * col("x_double") + intercept + rand() * noise_std, 2))
)

training_data = data.rdd.map(lambda row: (row.y, Vectors.dense([row.x_double]))).toDF(
    ["label", "features"]
)

# Create a LinearRegression model
lr = LinearRegression(featuresCol="features", labelCol="label")

lr_model = lr.fit(training_data)

print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

predictions = lr_model.transform(training_data)
predictions.show()

# evaluate
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2"
)
r2 = evaluator.evaluate(predictions)
print("R-squared (R2) on training data = %g" % r2)

# Stop the SparkSession
spark.stop()
