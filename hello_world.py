from pyspark.sql import SparkSession


def hello_world_pyspark():
    """A simple PySpark application that prints 'Hello, World!'."""

    # Create a SparkSession
    spark = SparkSession.builder.appName("HelloWorldApp").getOrCreate()

    # Simple RDD
    data = ["Hello, World!"]
    rdd = spark.sparkContext.parallelize(data)

    for line in rdd.collect():
        print(line)

    # Stop the SparkSession
    spark.stop()


if __name__ == "__main__":
    hello_world_pyspark()
