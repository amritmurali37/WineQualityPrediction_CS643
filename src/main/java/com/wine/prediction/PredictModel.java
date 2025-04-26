package com.wine.prediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class PredictModel {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: PredictModel <validation_data>");
            System.exit(1);
        }

        String validationPath = args[0];

        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .getOrCreate();

        Dataset<Row> validationData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ",")  
                .load(validationPath);

        String labelCol = "quality";
        String[] featureCols = validationData.columns();
        featureCols = java.util.Arrays.stream(featureCols)
                .filter(c -> !c.equals(labelCol))
                .toArray(String[]::new);

        // next line will cast all feature columns and label column to DoubleType
        for (String colName : featureCols) {
            validationData = validationData.withColumn(colName, validationData.col(colName).cast("double"));
        }
        validationData = validationData.withColumn(labelCol, validationData.col(labelCol).cast("double"));

        PipelineModel model = PipelineModel.load("wine_model");

        Dataset<Row> predictions = model.transform(validationData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);

        System.out.println("F1 Score = " + f1);

        spark.stop();
    }
}
