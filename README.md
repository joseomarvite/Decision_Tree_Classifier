# Decision Tree Classifier



# Introduction

A decision tree is a series of nodes, a directional graph that starts at the base with a single node and extends to the many leaf nodes that represent the categories that the tree can classify. Another way to think of a decision tree is as a flow chart, where the flow starts at the root node and ends with a decision made at the leaves. It is a decision-support tool. It uses a tree-like graph to show the predictions that result from a series of feature-based splits.

## Decision Tree Classifier

Here are some useful terms for describing a decision tree: 

![enter image description here](https://skymind.ai/images/wiki/decision_tree.png)

 - Root Node: A root node is at the beginning of a tree. It represents  entire population being analyzed. From the root node, the population  is divided according to various features, and those sub-groups are split in turn at each decision node under the root node. 
 -   Splitting: It is a process of dividing a node into two or more sub-nodes.
 
 -   Decision Node: When a sub-node splits into further sub-nodes, it’s a decision node.
 
 - Leaf Node or Terminal Node: Nodes that do not split are called leaf or terminal nodes.
 
 - Pruning: Removing the sub-nodes of a parent node is called pruning. A tree is grown through splitting and shrunk through pruning.
 
 -   Branch or Sub-Tree: A sub-section of decision tree is called branch or a sub-tree, just as a portion of a graph is called a sub-graph.
 
 -   Parent Node and Child Node: These are relative terms. Any node that falls under another node is a child node or sub-node, and any node which precedes those child nodes is called a parent node.

![enter image description here](https://skymind.ai/images/wiki/decision_tree_nodes.png)

## Decision trees are a popular algorithm for several reasons:

-   Explanatory Power: The output of decision trees is interpretable. It can be understood by people without analytical or mathematical backgrounds. It does not require any statistical knowledge to interpret them.

-   Exploratory data analysis: Decision trees can enable analysts to identify significant variables and important relations between two or more variables, helping to surface the signal contained by many input variables.

-   Minimal data cleaning: Because decision trees are resilient to outliers and missing values, they require less data cleaning than some other algorithms.

-   Any data type: Decision trees can make classifications based on both numerical and categorical variables.

## **Disadvantages**

-   Overfitting: Over fitting is a common flaw of decision trees. Setting constraints on model parameters (depth limitation) and making the model simpler through pruning are two ways to regularize a decision tree and improve its ability to generalize onto the test set.

-   Predicting continuous variables: While decision trees can ingest continuous numerical input, they are not a practical way to predict such values, since decision-tree predictions must be separated into discrete categories, which results in a loss of information when applying the model to continuous values.

-   Heavy feature engineering: The flip side of a decision tree’s explanatory power is that it requires heavy feature engineering. When dealing with unstructured data or data with latent factors, this makes decision trees sub-optimal. Neural networks are clearly superior in this regard.

## Source
[https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier)

## Code
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Se cargan los datos en la variable "data" en el formato "libsvm"
// El archivo a cargar obligatoriamente debe estar estructurado al formato
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Se agrega otra columna de indices, donde se tomaron los datos de la columna "label" y se 
// se transformaron a datos numericos, para poder manipularlos
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) // features with > 4 distinct values are treated as continuous.

// Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
// los datos que seran evaluados posteriormente para ver la precisión del arbol,
// el reparto de estos datos seran de forma aleatoria
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Se declara el Clasificador de árbol de decisión y se le agrega la columna que sera las etiquetas (indices) y
// los valores que cada respectivo indice (caracteristicas)
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

//
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Se entrena el modelo con los datos del arreglo "trainingData" que es el 70% de los datos totales
val model = pipeline.fit(trainingData)

// Se hacen las predicciones al tomar los datos sobrantes que se llevo "testData" que es el 30%
val predictions = model.transform(testData)

// Se manda a imprimir la etiqueta, sus respectivos valores y la prediccion de la etiqueta
predictions.select("predictedLabel", "label", "features").show(5)

// 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
// La variable "accuracy" tomara la acertación que hubo respecto a "predictedLabel" y "label"
val accuracy = evaluator.evaluate(predictions)
// Se manda a imprimir el resultado de error con respecto a la exactitud
println(s"Test Error = ${(1.0 - accuracy)}")


val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
// Se imprime el arbol de decisiones  
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```
## Outputs
```scala
scala> :load Decision_tree_classifier.scala
Loading Decision_tree_classifier.scala...
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
19/11/10 08:40:05 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_10d853c957b9
featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_f564b9c56b8a
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_f4a8946aecbb
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_0f5fe95ee967
pipeline: org.apache.spark.ml.Pipeline = pipeline_c049da4ed20b
model: org.apache.spark.ml.PipelineModel = pipeline_c049da4ed20b
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[95,96,97,12...|
|           0.0|  0.0|(692,[98,99,100,1...|
|           0.0|  0.0|(692,[122,123,124...|
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           1.0|  0.0|(692,[125,126,127...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[129,130,131...|
|           0.0|  0.0|(692,[152,153,154...|
|           0.0|  0.0|(692,[152,153,154...|
|           0.0|  0.0|(692,[153,154,155...|
|           1.0|  0.0|(692,[154,155,156...|
|           0.0|  0.0|(692,[181,182,183...|
|           1.0|  1.0|(692,[119,120,121...|
|           1.0|  1.0|(692,[123,124,125...|
+--------------+-----+--------------------+
only showing top 20 rows

evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_37cc9b54219c
accuracy: Double = 0.9142857142857143
Test Error = 0.08571428571428574
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_f4a8946aecbb) of depth 2 with 5 nodes
Learned classification tree model:
 DecisionTreeClassificationModel (uid=dtc_f4a8946aecbb) of depth 2 with 5 nodes
  If (feature 405 <= 21.0)
   If (feature 99 in {2.0})
    Predict: 0.0
   Else (feature 99 not in {2.0})
    Predict: 1.0
  Else (feature 405 > 21.0)
   Predict: 0.0


scala>
```
