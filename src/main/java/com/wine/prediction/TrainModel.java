package com.wine.prediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PipelineModel;

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
                .option("delimiter", ",")  // ✅ Comma delimiter
                .load(trainingPath);

        String labelCol = "quality";
        String[] featureCols = trainingData.columns();
        featureCols = java.util.Arrays.stream(featureCols)
                .filter(c -> !c.equals(labelCol))
                .toArray(String[]::new);

        // Cast all feature columns and label column to DoubleType
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

        model.save("wine_model");

        spark.stop();
    }
}
