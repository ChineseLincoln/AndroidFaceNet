package org.unreal.face.mtcnn

import android.content.res.AssetManager
import android.graphics.*
import android.util.Log
import java.util.*
import android.graphics.BitmapFactory
import kotlin.math.max
import kotlin.math.min


object PicUtils {
    //复制图片，并设置isMutable=true
    fun copyBitmap(bitmap: Bitmap): Bitmap {
        return bitmap.copy(bitmap.config, true)
    }

    //在bitmap中画矩形
    fun drawRect(bitmap: Bitmap, rect: Rect) {
        try {
            val canvas = Canvas(bitmap)
            val paint = Paint()
            val r = 255//(int)(Math.random()*255);
            val g = 0//(int)(Math.random()*255);
            val b = 0//(int)(Math.random()*255);
            paint.color = Color.rgb(r, g, b)
            paint.strokeWidth = (1 + bitmap.width / 500).toFloat()
            paint.style = Paint.Style.STROKE
            canvas.drawRect(rect, paint)
        } catch (e: Exception) {
            Log.i("Utils", "[*] error$e")
        }

    }

    //在图中画点
    fun drawPoints(bitmap: Bitmap, landmark: Array<Point?>) {
        for (i in landmark.indices) {
            val x = landmark[i]?.x?:0
            val y = landmark[i]?.y?:0
            //Log.i("Utils","[*] landmarkd "+x+ "  "+y);
            drawRect(bitmap, Rect(x - 1, y - 1, x + 1, y + 1))
        }
    }

    //Flip alone diagonal
    //对角线翻转。data大小原先为h*w*stride，翻转后变成w*h*stride
    fun flipDiag(data: FloatArray, h: Int, w: Int, stride: Int) {
        val tmp = FloatArray(w * h * stride)
        for (i in 0 until w * h * stride) tmp[i] = data[i]
        for (y in 0 until h)
            for (x in 0 until w) {
                for (z in 0 until stride)
                    data[(x * h + y) * stride + z] = tmp[(y * w + x) * stride + z]
            }
    }

    //src转为二维存放到dst中
    fun expand(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices)
            for (x in 0 until dst[0].size)
                dst[y][x] = src[idx++]
    }

    //src转为三维存放到dst中
    fun expand(src: FloatArray, dst: Array<Array<FloatArray>>) {
        var idx = 0
        for (y in dst.indices)
            for (x in 0 until dst[0].size)
                for (c in 0 until dst[0][0].size)
                    dst[y][x][c] = src[idx++]

    }

    //dst=src[:,:,1]
    fun expandProb(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices)
            for (x in 0 until dst[0].size)
                dst[y][x] = src[idx++ * 2 + 1]
    }

    //box转化为rect
    fun boxes2rects(boxes: Vector<Box>): Array<Rect?> {
        var cnt = 0
        for (i in 0 until boxes.size) if (!boxes.get(i).deleted) cnt++
        val r = arrayOfNulls<Rect>(cnt)
        var idx = 0
        for (i in 0 until boxes.size)
            if (!boxes.get(i).deleted)
                r[idx++] = boxes.get(i).transform2Rect()
        return r
    }

    //删除做了delete标记的box
    fun updateBoxes(boxes: Vector<Box>): Vector<Box> {
        val b = Vector<Box>()
        for (i in 0 until boxes.size)
            if (!boxes[i].deleted)
                b.addElement(boxes[i])
        return b
    }

    //
    fun showPixel(v: Int) {
        Log.i("MainActivity", "[*]Pixel:R" + (v shr 16 and 0xff) + "G:" + (v shr 8 and 0xff) + " B:" + (v and 0xff))
    }

    fun getBitmapFromAssets(assets: AssetManager?, fileName: String): Bitmap {
        val inputStream = assets?.open(fileName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream?.close()
        return bitmap
    }


    //按照rect的大小裁剪出人脸
    fun crop(bitmap: Bitmap, rect: Rect): Bitmap {
        return Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    }

    //rect上下左右各扩展pixels个像素
    fun rectExtend(bitmap: Bitmap, rect: Rect, pixels: Int) {
        rect.left = max(0, rect.left - pixels)
        rect.right = min(bitmap.width - 1, rect.right + pixels)
        rect.top = max(0, rect.top - pixels)
        rect.bottom = min(bitmap.height - 1, rect.bottom + pixels)
    }

}