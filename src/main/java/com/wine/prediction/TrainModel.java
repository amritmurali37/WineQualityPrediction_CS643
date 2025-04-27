package com.wine.prediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

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

        // Cast feature columns and label column to DoubleType
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

        // Save the model
        model.save("wine_model");

        // Evaluate the model
        Dataset<Row> predictions = model.transform(trainingData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(labelCol)
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 Score = " + f1Score);

        // Write F1 Score to a text file
        try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream("/home/hadoop/F1_Score.txt"), StandardCharsets.UTF_8))) {
            writer.println("F1 Score = " + f1Score);
        } catch (Exception e) {
            e.printStackTrace();
        }

        spark.stop();
    }
}
