package narendra

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{DoubleType, DataType}
import org.apache.spark.sql.{Row, Dataset, functions, SparkSession}

/**
 * Created by parmana on 9/23/17.
 */
object Titanic2ndApproach extends App{

  val spark = SparkSession.builder().appName("myML").master("local").getOrCreate()
  import spark.implicits._
  import functions._

  val convert = (str:String) =>  {
    val res = {
      if(str.equalsIgnoreCase("male"))
        1
      else
        0
    }
    res
  }
  import org.apache.spark.sql.functions.udf
  val convertUDF = udf(convert)
  val ds: Dataset[Row]  = spark.read.option("header","true").
    csv("src/test/titanic/train.csv").select(col("age").cast(DoubleType),col("Survived").cast(DoubleType),col("sex"))
    .filter(col("age").isNotNull).withColumn("gender",convertUDF(col("sex"))).drop("sex")


  val testDS: Dataset[Row]  = spark.read.option("header","true").
    csv("src/test/titanic/test.csv").select(col("age").cast(DoubleType),col("PassengerId").cast(DoubleType),col("sex"))
    .filter(col("age").isNotNull).withColumn("gender",convertUDF(col("sex"))).drop("sex")

  val labledata= spark.read.option("header","true").
    csv("src/test/titanic/gender_submission.csv").select(col("PassengerId").cast(DoubleType),col("Survived").cast(DoubleType))

  //labledata.show(false)
   //ds.show(false)
   //ds.printSchema()
  val featureCols = Array("age", "gender")
  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val df2 = assembler.transform(ds)


  val testAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val testDS2 = testAssembler.transform(testDS)



  //df2.show(false)
  val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("label")
  val df3 = labelIndexer.fit(df2).transform(df2)
  //df3.show(false)
  //val splitSeed = 5043
 // val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)
  val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.1).setElasticNetParam(0.1)
  val model = lr.fit(df3)

  //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")
  val predictions = model.transform(testDS2)

  predictions.show(false)
//  predictions.select(col("PassengerId"),col("prediction")).show(false)
  predictions.printSchema()
   //println( "predictions" +predictions.count() +" labledata "+labledata.count())
 val finaldata= predictions.join(labledata,"PassengerId").select(col("PassengerId"),col("prediction"),col("Survived")).show(false)

  /*val lp = predictions.select( "Survived", "prediction")
  val counttotal = predictions.count()
  val correct = lp.filter($"Survived" === $"prediction").count()
  val wrong = lp.filter(not($"Survived" === $"prediction")).count()
  val truep = lp.filter($"prediction" === 0.0).filter($"Survived" === $"prediction").count()
  val falseN = lp.filter($"prediction" === 0.0).filter(not($"Survived" === $"prediction")).count()
  val falseP = lp.filter($"prediction" === 1.0).filter(not($"Survived" === $"prediction")).count()
  val ratioWrong=wrong.toDouble/counttotal.toDouble
  val ratioCorrect=correct.toDouble/counttotal.toDouble*/


}
