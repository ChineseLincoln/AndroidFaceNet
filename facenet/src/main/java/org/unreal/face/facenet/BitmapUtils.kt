package org.unreal.face.facenet

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import kotlin.math.max
import kotlin.math.min


object BitmapUtils{

    fun scaleImage(inputSize:Long ,bitmap: Bitmap): Bitmap {
        val scaleWidth = (inputSize.toFloat() / bitmap.width.toFloat())
        val scaleHeight = (inputSize.toFloat() / bitmap.height.toFloat())
        println("scaleWidth ======= ${scaleWidth}")
        println("scaleHeight ======= ${scaleHeight}")
        val matrix = Matrix()
        matrix.postScale(scaleWidth ,scaleHeight)
        return BitmapUtils.createNewBitmap(bitmap, matrix)
    }

    private fun createNewBitmap(source: Bitmap, matrix : Matrix): Bitmap {
        return Bitmap.createBitmap(source , 0 , 0 , source.width , source.height , matrix , true)
    }

}