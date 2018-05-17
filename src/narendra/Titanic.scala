package narendra

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{functions, Dataset, Row, SparkSession}

/**
 * Created by parmana on 9/22/17.
 */
object Titanic extends App{


  val spark = SparkSession.builder().appName("myML").master("local").getOrCreate()
  import spark.implicits._
  import functions._

  val ds: Dataset[Row]  = spark.read.option("header","true").
    csv("src/test/titanic/train.csv").filter(col("age").isNotNull)

  val test: Dataset[Row]  = spark.read.option("header","true").
    csv("src/test/titanic/test.csv").filter(col("age").isNotNull)


  ds.show(false)

  val input= ds.select(col("Survived"),col("sex"),col("age")).map(line=>{
     val s= if(line.getString(1).equalsIgnoreCase("male")){
       1
     } else{
       2
     }
    val features = Vectors.dense(s.toDouble, line.getString(2).toDouble)
    LabeledPoint(line.getString(0).toDouble,features)
  })

  val testdata= test.select(col("PassengerId"),col("sex"),col("age")).map(line=>{
    val s= if(line.getString(1).equalsIgnoreCase("male")){
      1
    } else{
      2
    }
    val features = Vectors.dense(s.toDouble, line.getString(2).toDouble)
    LabeledPoint(line.getString(0).toDouble,features)
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
  val output = model.transform(testdata)
  println(s"output schema ${output.printSchema()}")

  output.foreach(
    (result) => {
      //println(s"predicted label: ${result(4)}, actual label: ${result(0)}")
      println(s"probability: ${result}")
    }
  )

  /*output.foreach(
    (result) => {
      println(s"predicted label: ${result(4)}, actual label: ${result(0)}")
      println(s"probability: ${result(3)}")
    }
  )*/

  /*df.map(line => {
    val features = Vectors.dense(line.getDouble(), arrayDouble(1))
  })*/

 //  ds.show(false)


 // val ds = spark.read.text("src/test/titanic/train.csv").as[String]
  //input.show(false)


  def my(string: String):Int={
    val s= if(string.equalsIgnoreCase("male")){
      1
    } else {
      0
    }
    s
  }


  //case class lb (serviedId:Double,lable:Double)

}
