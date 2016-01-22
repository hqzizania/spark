/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.nlp

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer
import riso.numerical.LBFGS

private[mllib] class CRF extends Serializable {
  private val freq: Int = 1   //TODO add new feature for freq > 1
  private val maxiter: Int = 100000
//  private val cost: Double = 1.0
  private val eta: Double = 0.0001
  private val C: Double = 1.0   //TODO C can be set by user
  private var featureIdx: FeatureIndex = new FeatureIndex()


  /**
   * Internal method to verify the CRF model
   * @param test the same source in the CRFLearn
   * @param model the output from CRFLearn
   * @return the source with predictive labels
   */
  def verify(test: Array[String],
             model: Array[String]): Array[String] = {
    var tagger: Tagger = new Tagger()
    // featureIdx = featureIdx.openTagSet(test)
    tagger = tagger.read(test)
    featureIdx = featureIdx.openFromArray(model)
    tagger.open(featureIdx)
    tagger.parse()
    tagger.createOutput()
  }

  /**
   * Internal method to train the CRF model
   * @param template the template to train the model
   * @param train the source for the training
   * @return the model of the source
   */
  def learn(template: Array[String],
            train: RDD[Array[String]]): Array[ArrayBuffer[String]] = {
    var taggerList: ArrayBuffer[Tagger] = new ArrayBuffer[Tagger]()
    featureIdx.openTemplate(template)

    var i = 0
    val trainC: Array[Array[String]] = train.toLocalIterator.toArray
    while (i < trainC.size) {
      featureIdx.openTagSet(trainC(i))
      i += 1
    }
    i = 0
    featureIdx.y = featureIdx.y.distinct
    while (i < trainC.size) {
      var tagger: Tagger = new Tagger()
      tagger.open(featureIdx)
      tagger = tagger.read(trainC(i))
      featureIdx.buildFeatures(tagger)
      if (tagger != null) taggerList.append(tagger)
      i += 1
    }
    featureIdx.shrink(freq)
    featureIdx.initAlpha(featureIdx.maxid)
    runCRF(taggerList, featureIdx, train.sparkContext)
    featureIdx.saveModel
  }

  /**
   * Parse segments in the unit sentences or paragraphs
   * @param tagger the tagger in the template
   * @param featureIndex the index of the feature
   */

  def runCRF(tagger: ArrayBuffer[Tagger], featureIndex: FeatureIndex,
             sc: SparkContext): Unit = {
    var diff: Double = 0.0
    var old_obj: Double = 1e37
    var converge: Int = 0
    var itr: Int = 0
    var all: Int = 0
//    val opt = new Optimizer()
    var i: Int = 0
    val iFlag: Array[Int] = Array(0)
    val diagH = Array.fill(featureIdx.maxid)(0.0)
    val iPrint = Array(-1,0)
    val xTol = 1.0E-16
//    while (i < tagger.length) {
//      all += tagger(i).x.size
//      i += 1
//    }
//    i = 0
    tagger.foreach(all += _.x.size)


    //    val taggers: RDD[Tagger] = sc.parallelize(tagger, 2)  //TODO add RDD[Tagger]

    while (itr < maxiter) {

      var x: ArrayBuffer[Tagger] = tagger
      var err: Int = 0
      var zeroOne: Int = 0
      var size: Int = 0
      var obj: Double = 0.0
      val expected: Array[Double] = Array.fill(featureIdx.maxid)(0.0)
      var expected3: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      expected.copyToBuffer(expected3)
//      var expected: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      var idx: Int = 0



//      initExpected()
      while (idx >= 0 && idx < tagger.size) {
        val expected: Array[Double] = Array.fill(featureIdx.maxid)(0.0)
        var expected2: ArrayBuffer[Double] = new ArrayBuffer[Double]()
        expected.copyToBuffer(expected2)
        obj += x(idx).gradient(expected2)
        val err_num = x(idx).eval()
        err += err_num
        if (err_num != 0) {
          zeroOne += 1
        }
        idx += 1

        var xx = 0
        while (xx < featureIdx.maxid){
          expected3(xx) += expected2(xx)
          xx += 1
        }
      }

      var k: Int = 0
      while (k < featureIndex.maxid) {
        obj += featureIdx.alpha(k) * featureIdx.alpha(k) / (2.0 * C)
        expected3(k) += featureIdx.alpha(k) / C
        k += 1
      }
      k = 0

      i = 0
      if (itr == 0) {
        diff = 1.0
      } else {
        diff = math.abs((old_obj - obj) / old_obj)
      }
      old_obj = obj
      printf("iter=%d, terr=%2.5f, serr=%2.5f, act=%d, obj=%2.5f,diff=%2.5f\n",
        itr, 1.0 * err / all,
        1.0 * zeroOne / tagger.size, featureIndex.maxid,
        obj, diff)
      if (diff < eta) {
        converge += 1
      } else {
        converge = 0
      }
      if (converge == 3) {
        itr = maxiter + 1 // break
      }




      LBFGS.lbfgs(featureIdx.maxid, 5, featureIdx.alpha, obj, expected3.toArray, false, diagH, iPrint, 1e-7, xTol, iFlag)
//      opt.optimizer(featureIndex.maxid, alpha, obj, expected3, C)


      itr += 1
    }
  }
}


@DeveloperApi
object CRF {
  @transient var sc: SparkContext = _

  /**
   * Train CRF Model
   * Feature file format
   * word|word characteristic|designated label
   *
   * @param templates Source templates for training the model
   * @param features Source files for training the model
   * @return Model of a unit
   */
  def runCRF(templates: RDD[Array[String]],
             features: RDD[Array[String]]): CRFModel = {
    val template = templates.toLocalIterator.toArray.flatten
    val crf = new CRF()
    val model = crf.learn(template, features)
    new CRFModel(model)
  }

  /**
   * Verify CRF model
   * Test result format:
   * word|word characteristic|designated label|predicted label
   *
   * @param tests  Source files to be verified
   * @param models Model files after call the CRF learn
   * @return Source files with the predictive labels
   */
  def verifyCRF(tests: RDD[Array[String]],
                models: RDD[Array[String]]): CRFModel = {
//    val test: Array[Array[String]] = tests.toLocalIterator.toArray
    val model: Array[String] = models.toLocalIterator.toArray.flatten
    sc = tests.sparkContext
    val finalArray: Array[ArrayBuffer[String]] = tests.map(x => {
      val crf = new CRF()
      val a = new ArrayBuffer[String]()
      val result: Array[String] = crf.verify(x, model)
      result.copyToBuffer(a)
      a
    }).collect()
    new CRFModel(finalArray)
  }

  /**
   * Get spark context
   * @return the current spark context
   */
  def getSparkContext: SparkContext = {
    if (sc != null) {
      sc
    } else {
      null
    }
  }
}
