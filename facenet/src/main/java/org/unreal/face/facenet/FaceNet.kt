package org.unreal.face.facenet

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class FaceNet(assetManager: AssetManager){

    private val modelFile = "facenet-20180408-102900.pb"
    private val inputNode = "input:0"
    private val outputNode = "embeddings:0"
    private val phaseName = "phase_train:0"

    private val TAG = "FaceNet"

    private val tensorFlowInference = TensorFlowInferenceInterface(assetManager , modelFile)

    private val inputSize = 160L

    private val floatValues = FloatArray((inputSize * inputSize * 3).toInt())
    private val intValues = IntArray((inputSize * inputSize).toInt())

    private val outputNames = arrayOf(outputNode)



    private fun normalizeImage(bitmap: Bitmap){
        val scaleImage = BitmapUtils.scaleImage(inputSize , bitmap)
        val imageMean = 127.5f
        val imageStd = 128
        scaleImage?.getPixels(intValues ,
            0,
            scaleImage.width ,
            0,
            0,
            scaleImage.width ,
            scaleImage.height)
        for (i in 0 until intValues.size ){
            val intVar = intValues[i]
            floatValues[i * 3 + 0] = ((intVar shr 16 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 1] = ((intVar shr 8 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 2] = ((intVar and 0xFF) - imageMean) / imageStd
        }
    }

    fun recognizeImage(bitmap: Bitmap): FaceFeature {
        //Log.d(TAG,"[*]recognizeImage");
        //(0)图片预处理，normailize
        normalizeImage(bitmap)
        //(1)Feed
        try {
            tensorFlowInference.feed(inputNode, floatValues, 1L, inputSize, inputSize, 3L)
            val phase = BooleanArray(1)
            phase[0] = false
            tensorFlowInference.feed(phaseName, phase)
        } catch (e: Exception) {
            Log.e(TAG, "[*] feed Error\n$e")
        }

        //(2)run
        // Log.d(TAG,"[*]Feed:"+INPUT_NAME);
        try {
            tensorFlowInference.run(outputNames, false)
        } catch (e: Exception) {
            Log.e(TAG, "[*] run error\n$e")
        }

        //(3)fetch
        val faceFeature = FaceFeature()
        val outputs = faceFeature.feature
        try {
            tensorFlowInference.fetch(outputNode, outputs)
        } catch (e: Exception) {
            Log.e(TAG, "[*] fetch error\n$e")
        }

        return faceFeature
    }
}