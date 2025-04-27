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
import software.amazon.awssdk.auth.credentials.InstanceProfileCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.FileWriter;
import java.nio.file.Paths;

public class TrainModel {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: TrainModel <training_data>");
            System.exit(1);
        }

        String trainingPath = args[0];
        String s3BucketName = "bucketforassigntwo";  // bucket created for outputs
        String s3Key = "F1_Score.txt";               // outputs the f1 - score
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

        model.write().overwrite().save("wine_model");

   
        Dataset<Row> predictions = model.transform(trainingData);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(labelCol)
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        
        // Save to local file
        FileWriter fw = new FileWriter("/tmp/F1_Score.txt");
        fw.write("F1 Score: " + f1Score);
        fw.close();

        // Upload to S3
        S3Client s3 = S3Client.builder()
                .region(Region.US_EAST_1)
                .credentialsProvider(InstanceProfileCredentialsProvider.create())
                .build();

        s3.putObject(PutObjectRequest.builder()
                        .bucket(s3BucketName)
                        .key(s3Key)
                        .build(),
                Paths.get("/tmp/F1_Score.txt"));

        System.out.println("F1 Score uploaded to S3!");

        spark.stop();
    }
}
