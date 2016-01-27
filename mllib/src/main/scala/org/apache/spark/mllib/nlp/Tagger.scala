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

import breeze.numerics.{exp, log}

import scala.collection.mutable.ArrayBuffer

private[mllib] class Tagger extends Serializable {
  var mode: Int = 2
  // LEARN
  var vlevel: Int = 0
  var nbest: Int = 0
  var ysize: Int = 0
  var cost: Double = 0.0
  var Z: Double = 0.0
  var feature_id: Int = 0
  var thread_id: Int = 0
  var feature_idx: FeatureIndex = new FeatureIndex()
  var x: ArrayBuffer[Array[String]] = new ArrayBuffer[Array[String]]()
  var node: ArrayBuffer[ArrayBuffer[Node]] = new ArrayBuffer[ArrayBuffer[Node]]
  var penalty: ArrayBuffer[ArrayBuffer[Double]] = new ArrayBuffer[ArrayBuffer[Double]]()
  var answer: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  var result: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  val MINUS_LOG_EPSILON = 50

  /**
   * Get the feature index
   */
  def open(featureIndex: FeatureIndex): Unit = {
    feature_idx = featureIndex
    ysize = feature_idx.y.size
  }

  /**
   * Read feature files, one RDD contains many features files.
   * Each feature is a string in the RDD. The feature should be correspondent
   * with template file. User needs prepare the relationship beforehand.
   */
  def read(lines: Array[String]): Tagger = {
    var i: Int = 0
    var columns: Array[String] = null
    var j: Int = 0
    while (i < lines.length) {
      if (lines(i).charAt(0) != '\0'
        && lines(i).charAt(0) != ' '
        && lines(i).charAt(0) != '\t') {
        columns = lines(i).split('|')
        x.append(columns)
        for(y <- feature_idx.y if y.equals(columns(feature_idx.xsize)))
          answer.append(feature_idx.y.indexOf(y))
        result.append(0)
      }
      i += 1
    }
    //TODO node_[s].resize(ysize_);
    this
  }

  def setFeatureId(id: Int): Unit = {
    feature_id = id
  }

  def shrink(): Unit = {
    feature_idx.buildFeatures(this)
  }

  /**
   * Build the matrix to calculate
   * cost of each node according to template
   */
  def buildLattice(): Unit = {
    var i: Int = 0
    var j: Int = 0
    var k: Int = 0

    if (x.nonEmpty) {
      feature_idx.rebuildFeatures(this)
      while (i < x.length) {
        while (j < ysize) {
          node(i)(j) = feature_idx.calcCost(node(i)(j))
//          printf( "node.cost = %2.5f\n", node(i)(j).cost)
          while (k < node(i)(j).lpath.length) {
            node(i)(j).lpath(k) = feature_idx.calcCost(node(i)(j).lpath(k))
//            printf( "path.cost = %2.5f\n", node(i)(j).lpath(k).cost)
            k += 1
          }
          k = 0
          j += 1
        }
        j = 0
        i += 1
      }
    }
//    i = 0
//    j = 0
//    if (penalty.nonEmpty) {
//      while (i < x.length) {
//        while (j < ysize) {
//          node(i)(j).cost += penalty(i)(j)
//          j += 1
//        }
//        j = 0
//        i += 1
//      }
//    }
  }

  /**
   *Calculate the expectation of each node
   * https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
   * http://www.cs.columbia.edu/~mcollins/fb.pdf
   */
  def forwardBackward(): Unit = {
    var idx: Int = x.length - 1
    var i: Int = 0
    var j: Int = 0
    if (x.nonEmpty) {
      while (i < x.length) {
        while (j < ysize) {
          node(i)(j).calcAlpha()
          j += 1
        }
        j = 0
        i += 1
      }
      j = 0
      while (idx >= 0) {
        while (j < ysize) {
          node(idx)(j).calcBeta()
          j += 1
        }
        j = 0
        idx -= 1
      }
      Z = 0.0
      i = 0
      while (i < ysize) {
        Z = logsumexp(Z, node(0)(i).beta, i == 0)
        i += 1
      }
    }
  }

  /**
   * Get the max expectation in the nodes and predicts the most likely label
   * http://www.cs.utah.edu/~piyush/teaching/structured_prediction.pdf
   * http://www.weizmann.ac.il/mathusers/vision/courses/2007_2/files/introcrf.pdf
   * Page 15
   */
  def viterbi(): Unit = {
    var bestc: Double = -1e37
    var best: Node = null
    var cost: Double = 0.0
    var nd: Node = null
    var i: Int = 0
    var j: Int = 0
    var k: Int = 0
    while (i < x.length) {
      while (j < ysize) {
        bestc = -1e37
        best = null
        while (k < node(i)(j).lpath.length) {
          cost = node(i)(j).lpath(k).lnode.bestCost + node(i)(j).lpath(k).cost + node(i)(j).cost
          if (cost > bestc) {
            bestc = cost
            best = node(i)(j).lpath(k).lnode
          }
          k += 1
        }
        node(i)(j).prev = best
        if (best != null) {
          node(i)(j).bestCost = bestc
        } else {
          node(i)(j).bestCost = node(i)(j).cost
        }
        k = 0
        j += 1
      }
      j = 0
      i += 1
    }
    bestc = -1e37
    best = null
    j = 0
    while (j < ysize) {
      if (node(x.length - 1)(j).bestCost > bestc) {
        best = node(x.length - 1)(j)
        bestc = node(x.length - 1)(j).bestCost
      }
      j += 1
    }
    nd = best
    while (nd != null) {
      result.update(nd.x, nd.y)
      nd = nd.prev
    }
    cost = -node(x.length - 1)(result(x.length - 1)).bestCost
  }

  def gradient(expected: ArrayBuffer[Double]): Double = {
    var s: Double = 0.0
    var lNode: Node = null
    var rNode: Node = null
    var lPath: Path = null
    var idx: Int = 0
    var i: Int = 0
    var j: Int = 0
    var row: Int = 0
    var rIdx: Int = 0

    if (x.isEmpty) {
      return 0.0
    }
    buildLattice()
    forwardBackward()

    while (i < x.length) {
      while (j < ysize) {
        node(i)(j).calExpectation(expected, Z, ysize, feature_idx)
//        printf("node(%d)(%d) = %2.5f\n", i, j, node(i)(j).cost)
        j += 1

      }
      j = 0
      i += 1
    }

    i = 0
    j = 0
    while (row < x.length) {
//      idx = node(row)(answer(row)).fvector
//      rIdx = feature_idx.getFeatureCacheIdx(node(row)(answer(row)).fvector)
      rIdx = node(row)(answer(row)).fvector
      while (feature_idx.featureCache(rIdx) != -1) {
        expected(feature_idx.featureCache(rIdx) + answer(row)) -= 1.0
        rIdx += 1
      }
//      rIdx = 0
      s += node(row)(answer(row)).cost
      while (i < node(row)(answer(row)).lpath.length) {
        lNode = node(row)(answer(row)).lpath(i).lnode
        rNode = node(row)(answer(row)).lpath(i).rnode
        lPath = node(row)(answer(row)).lpath(i)
        if (lNode.y == answer(lNode.x)) {
//          idx = lPath.fvector
//          rIdx = feature_idx.getFeatureCacheIdx(lPath.fvector)
          rIdx = lPath.fvector
          while (feature_idx.featureCache(rIdx) != -1) {
            expected(feature_idx.featureCache(rIdx) + lNode.y * ysize + rNode.y) -= 1.0
            rIdx += 1
          }
          s += lPath.cost
        }
        i += 1
      }
      i = 0
      row += 1
    }
    viterbi()
    Z - s
  }

  def eval(): Int = {
    var err: Int = 0
    var i: Int = 0
    while (i < x.length) {
      if (answer(i) != result(i)) {
        err += 1
      }
//      printf("answer[%d] = %d, result[%d] = %d\n", i, answer(i), i, result(i))
      i += 1
    }
    err
  }

  /**
   * simplify the log likelihood.
   */
  def logsumexp(x: Double, y: Double, flg: Boolean): Double = {
    if (flg) return y
    val vMin: Double = math.min(x, y)
    val vMax: Double = math.max(x, y)
    if (vMax > vMin + MINUS_LOG_EPSILON) {
      vMax
    } else {
      vMax + log(exp(vMin - vMax) + 1.0)
    }
  }

  def getFeatureIdx(): FeatureIndex = {
    if (feature_idx != null) {
      return feature_idx
    }
    null
  }

  def parse(): Unit = {
    buildLattice()
    if (nbest != 0 || vlevel >= 1) {
      forwardBackward()
    }
    viterbi()
  }

  def createOutput(): Array[String] = {
    var i: Int = 0
    var j: Int = 0
    var content: ArrayBuffer[String] = new ArrayBuffer[String]
    while (i < x.size) {
      while (j < x(i).length) {
        if(j == x(i).length - 1){
          content.append(x(i)(j))
        }
        else
          content.append(x(i)(j) + "|")
        j += 1
      }
      content += feature_idx.y(result(i))
      i += 1
      j = 0
    }
    content.toArray
  }

}
