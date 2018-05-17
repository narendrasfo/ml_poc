import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.regression.{LinearRegressionSummary, LinearRegressionModel, LinearRegression}

/**
 * Created by Pankaj on 9/13/17.
 */
object MyLinearRegression {

  def main(args: Array[String]) = {

    val spark = SparkSession.builder().appName("myML").master("local").getOrCreate()

    runLinearRegression(spark)

  }

  def runLinearRegression(spark: SparkSession): Unit = {

    import spark.implicits._
    val ds =  spark.read.text("src/test/data_linear_regression").as[String]
    val arr = ds.take(5)

    println(s"rddcount ${ds.count}")
    arr.foreach( line => {

      println(s"${line}")
    })

    val parsedData = ds.map { line =>
      val x : Array[String] = line.replace(",", " ").split(" ")
      val y = x.map (
        a => {
          a.toDouble
        }
      )

      val d = y.size - 1
      val c = Vectors.dense(y(1),y(2))

      println(s"c => $c")
      LabeledPoint(y(0), c)
    }.cache()

    val numIterations = 1000
    val alpha = .01
    val lambda = .01
    val lr = new LinearRegression()
    //set alpha
    lr.setElasticNetParam(alpha)

    //set lambda
    lr.setRegParam(lambda)

    //set number of iterations
    lr.setMaxIter(numIterations)

    val model = lr.fit(parsedData)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val output: DataFrame = model.transform(parsedData)

    output.foreach(
      (result) => println(s"predicted label: ${result(0)}, actual label: ${result(2)}"))

  }

}