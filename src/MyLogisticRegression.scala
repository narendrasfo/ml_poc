import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * Created by Pankaj on 9/22/17.
 */
object MyLogisticRegression {

  def main(args: Array[String]) = {

    val spark = SparkSession.builder().appName("myML").master("local").getOrCreate()

    runLogisticRegression(spark)
  }

  def runLogisticRegression(spark: SparkSession) = {

    import spark.implicits._

    val ds = spark.read.text("src/test/data_logistic_regression").as[String]

    val input = ds.map( line => {

      val array = line.split(" ")
      val arrayDouble = array.map( x => x.toDouble)
      val features = Vectors.dense(arrayDouble(0), arrayDouble(1))
      LabeledPoint(arrayDouble(2),features)
    })

    
    val numIterations = 1000
    val alpha = .01
    val lambda = .01

    val lr = new LogisticRegression()

    lr.setMaxIter(numIterations)
    lr.setElasticNetParam(alpha)
    lr.setRegParam(lambda)

    val model = lr.fit(input)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")


    val output = model.transform(input)
    println(s"output schema ${output.printSchema()}")
    output.foreach(
      (result) => {
        println(s"predicted label: ${result(4)}, actual label: ${result(0)}")
        println(s"probability: ${result(3)}")
      }
    )

  }
}
