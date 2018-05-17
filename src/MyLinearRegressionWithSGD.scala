import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.sql._

/**
 * Created by Pankaj on 9/21/17.
 */
object MyLinearRegressionWithSGD {

  def main(args: Array[String]) = {

    val spark = SparkSession.builder().appName("myML").master("local").getOrCreate()

    runLinearRegression(spark)

  }

  def runLinearRegression(spark: SparkSession): Unit = {

    import spark.implicits._
    val ds = spark.read.text("src/test/data_linear_regression").as[String]
    val arr = ds.take(5)

    println(s"rddcount ${ds.count}")
    arr.foreach(line => {

      println(s"${line}")
    })

    val parsedData = ds.map { line =>
      val x: Array[String] = line.replace(",", " ").split(" ")
      val y = x.map(
        a => {
          a.toDouble
        }
      )

      val d = y.size - 1
      val c = Vectors.dense(y(1), y(2))

      println(s"c => $c")
      LabeledPoint(y(0), c)
    }.cache()

    val numIterations = 1000
    val alpha = .01
    val lambda = .01

    val model = LinearRegressionWithSGD.train(parsedData.rdd, numIterations, alpha)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${model.weights} Intercept: ${model.intercept}")


    // for LinearRegressionWithSGD
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    valuesAndPreds.foreach(

      (result) => println(s"predicted label: ${result._1}, actual label: ${result._2}")
    )

  }
}
