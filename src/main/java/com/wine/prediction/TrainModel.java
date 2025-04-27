package com.wine.prediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PipelineModel;

import java.io.FileWriter;

public class TrainModel {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: TrainModel <training_data>");
            System.exit(1);
        }

        String trainingPath = args[0];

        SparkSession spark = SparkSession.builder()
                .appName("WineQualityTraining")
                .getOrCreate();

        Dataset<Row> trainingData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ",")
                .load(trainingPath);

        String labelCol = "quality";
        String[] featureCols = trainingData.columns();
        featureCols = java.util.Arrays.stream(featureCols)
                .filter(c -> !c.equals(labelCol))
                .toArray(String[]::new);

        for (String colName : featureCols) {
            trainingData = trainingData.withColumn(colName, trainingData.col(colName).cast("double"));
        }
        trainingData = trainingData.withColumn(labelCol, trainingData.col(labelCol).cast("double"));

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        LogisticRegression lr = new LogisticRegression()
                .setLabelCol(labelCol)
                .setFeaturesCol("features")
                .setMaxIter(100);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, lr});

        PipelineModel model = pipeline.fit(trainingData);

        // Save model
        model.write().overwrite().save("wine_model");

        // Evaluate model
        Dataset<Row> predictions = model.transform(trainingData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(labelCol)
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);

        System.out.println("F1 Score = " + f1);

        // Write F1 score to a text file
        FileWriter writer = new FileWriter("/home/hadoop/F1_Score.txt");
        writer.write("F1 Score = " + f1);
        writer.close();

        spark.stop();
    }
}
