package org.unreal.face.mtcnn

import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Point
import android.util.Log
import java.util.*
import kotlin.math.max
import kotlin.math.min


class MTCNN(assetManager: AssetManager){
    private val factor = 0.709f
    private val pNetThreshold = 0.6f
    private val rNetThreshold = 0.7f
    private val outputNetThreshold = 0.7f
    //MODEL PATH
    private val modelFile = "mtcnn_freezed_model.pb"
    //tensor name
    private val pNetInName = "pnet/input:0"
    private val pNetOutName = arrayOf("pnet/prob1:0", "pnet/conv4-2/BiasAdd:0")
    private val rNetInName = "rnet/input:0"
    private val rNetOutName = arrayOf("rnet/prob1:0", "rnet/conv5-2/conv5-2:0")
    private val outputNetInName = "onet/input:0"
    private val outputNetOutName = arrayOf("onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0")

    var lastProcessTime: Long = 0   //最后一张图片处理的时间ms
    private var inferenceInterface: TensorFlowInferenceInterface = TensorFlowInferenceInterface(assetManager, modelFile)

    private val TAG = "MTCNN"


    //读取Bitmap像素值，预处理(-127.5 /128)，转化为一维数组返回
    private fun normalizeImage(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val floatValues = FloatArray(w * h * 3)
        val intValues = IntArray(w * h)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val imageMean = 127.5f
        val imageStd = 128f

        for (i in intValues.indices) {
            val `val` = intValues[i]
            floatValues[i * 3 + 0] = ((`val` shr 16 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 1] = ((`val` shr 8 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 2] = ((`val` and 0xFF) - imageMean) / imageStd
        }
        return floatValues
    }

    /*
       检测人脸,minSize是最小的人脸像素值
     */
    private fun bitmapResize(bm: Bitmap, scale: Float): Bitmap {
        val width = bm.width
        val height = bm.height
        // CREATE A MATRIX FOR THE MANIPULATION。matrix指定图片仿射变换参数
        val matrix = Matrix()
        // RESIZE THE BIT MAP
        matrix.postScale(scale, scale)
        return Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, true
        )
    }

    //输入前要翻转，输出也要翻转
    private fun proposalNetForward(
        bitmap: Bitmap,
        pNetOutProb: Array<FloatArray>,
        pNetOutBias: Array<Array<FloatArray>>
    ): Int {
        val w = bitmap.width
        val h = bitmap.height

        val pNetIn = normalizeImage(bitmap)
        PicUtils.flipDiag(pNetIn, h, w, 3) //沿着对角线翻转
        inferenceInterface.feed(pNetInName, pNetIn, 1, w.toLong(), h.toLong(), 3)
        inferenceInterface.run(pNetOutName, false)
        val pNetOutSizeW = Math.ceil(w * 0.5 - 5).toInt()
        val pNetOutSizeH = Math.ceil(h * 0.5 - 5).toInt()
        val pNetOutP = FloatArray(pNetOutSizeW * pNetOutSizeH * 2)
        val pNetOutB = FloatArray(pNetOutSizeW * pNetOutSizeH * 4)
        inferenceInterface.fetch(pNetOutName[0], pNetOutP)
        inferenceInterface.fetch(pNetOutName[1], pNetOutB)
        //【写法一】先翻转，后转为2/3维数组
        PicUtils.flipDiag(pNetOutP, pNetOutSizeW, pNetOutSizeH, 2)
        PicUtils.flipDiag(pNetOutB, pNetOutSizeW, pNetOutSizeH, 4)
        PicUtils.expand(pNetOutB, pNetOutBias)
        PicUtils.expandProb(pNetOutP, pNetOutProb)
        /*
        *【写法二】这个比较快，快了3ms。意义不大，用上面的方法比较直观
        for (int y=0;y<pNetOutSizeH;y++)
            for (int x=0;x<pNetOutSizeW;x++){
               int idx=pNetOutSizeH*x+y;
               pNetOutProb[y][x]=pNetOutP[idx*2+1];
               for(int i=0;i<4;i++)
                   pNetOutBias[y][x][i]=pNetOutB[idx*4+i];
            }
        */
        return 0
    }

    //Non-Maximum Suppression
    //nms，不符合条件的deleted设置为true
    private fun nms(boxes: Vector<Box>, threshold: Float, method: String) {
        //NMS.两两比对
        //int delete_cnt=0;
        val cnt = 0
        for (i in 0 until boxes.size) {
            val box = boxes[i]
            if (!box.deleted) {
                //score<0表示当前矩形框被删除
                for (j in i + 1 until boxes.size) {
                    val box2 = boxes.get(j)
                    if (!box2.deleted) {
                        val x1 = max(box.box[0], box2.box[0])
                        val y1 = max(box.box[1], box2.box[1])
                        val x2 = min(box.box[2], box2.box[2])
                        val y2 = min(box.box[3], box2.box[3])
                        if (x2 < x1 || y2 < y1) continue
                        val areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1)
                        var iou = 0f
                        if (method == "Union")
                            iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU)
                        else if (method == "Min") {
                            iou = 1.0f * areaIoU / min(box.area(), box2.area())
                            Log.i(TAG, "[*]iou=$iou")
                        }
                        if (iou >= threshold) { //删除prob小的那个框
                            if (box.score > box2.score)
                                box2.deleted = true
                            else
                                box.deleted = true
                            //delete_cnt++;
                        }
                    }
                }
            }
        }
        //Log.i(TAG,"[*]sum:"+boxes.size+" delete:"+delete_cnt);
    }

    private fun generateBoxes(
        prob: Array<FloatArray>,
        bias: Array<Array<FloatArray>>,
        scale: Float,
        threshold: Float,
        boxes: Vector<Box>
    ): Int {
        val h = prob.size
        val w = prob[0].size
        //Log.i(TAG,"[*]height:"+prob.length+" width:"+prob[0].length);
        for (y in 0 until h)
            for (x in 0 until w) {
                val score = prob[y][x]
                //only accept prob >threadshold(0.6 here)
                if (score > threshold) {
                    val box = Box()
                    //score
                    box.score = score
                    //box
                    box.box[0] = Math.round(x * 2 / scale)
                    box.box[1] = Math.round(y * 2 / scale)
                    box.box[2] = Math.round((x * 2 + 11) / scale)
                    box.box[3] = Math.round((y * 2 + 11) / scale)
                    //bbr
                    for (i in 0..3)
                        box.bbr[i] = bias[y][x][i]
                    //add
                    boxes.addElement(box)
                }
            }
        return 0
    }

    private fun boundingBoxRegression(boxes: Vector<Box>) {
        for (i in 0 until boxes.size)
            boxes[i].calibrate()
    }

    //Pnet + Bounding Box Regression + Non-Maximum Regression
    /* NMS执行完后，才执行Regression
     * (1) For each scale , use NMS with threshold=0.5
     * (2) For all candidates , use NMS with threshold=0.7
     * (3) Calibrate Bounding Box
     * 注意：CNN输入图片最上面一行，坐标为[0..width,0]。所以Bitmap需要对折后再跑网络;网络输出同理.
     */
    private fun proposalNet(bitmap: Bitmap, minSize: Int): Vector<Box> {
        val whMin = min(bitmap.width, bitmap.height)
        var currentFaceSize = minSize.toFloat()  //currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
        val totalBoxes = Vector<Box>()
        //【1】Image Paramid and Feed to Pnet
        while (currentFaceSize <= whMin) {
            val scale = 12.0f / currentFaceSize
            //(1)Image Resize
            val bm = bitmapResize(bitmap, scale)
            val w = bm.width
            val h = bm.height
            //(2)RUN CNN
            val pNetOutSizeW = (Math.ceil(w * 0.5 - 5) + 0.5).toInt()
            val pNetOutSizeH = (Math.ceil(h * 0.5 - 5) + 0.5).toInt()
            val pNetOutProb = Array(pNetOutSizeH) { FloatArray(pNetOutSizeW) }
            val pNetOutBias = Array(pNetOutSizeH) { Array(pNetOutSizeW) { FloatArray(4) } }
            proposalNetForward(bm, pNetOutProb, pNetOutBias)
            //(3)数据解析
            val curBoxes = Vector<Box>()
            generateBoxes(pNetOutProb, pNetOutBias, scale, pNetThreshold, curBoxes)
            //Log.i(TAG,"[*]CNN Output Box number:"+curBoxes.size+" Scale:"+scale);
            //(4)nms 0.5
            nms(curBoxes, 0.5f, "Union")
            //(5)add to totalBoxes
            for (i in 0 until curBoxes.size)
                if (!curBoxes[i].deleted)
                    totalBoxes.addElement(curBoxes[i])
            //Face Size等比递增
            currentFaceSize /= factor
        }
        //NMS 0.7
        nms(totalBoxes, 0.7f, "Union")
        //BBR
        boundingBoxRegression(totalBoxes)
        return PicUtils.updateBoxes(totalBoxes)
    }

    //截取box中指定的矩形框(越界要处理)，并resize到size*size大小，返回数据存放到data中。
    var tmp_bm: Bitmap? = null

    private fun cropAndResize(bitmap: Bitmap, box: Box, size: Int, data: FloatArray) {
        //(2)crop and resize
        val matrix = Matrix()
        val scale = 1.0f * size / box.width()
        matrix.postScale(scale, scale)
        val croped = Bitmap.createBitmap(bitmap, box.left(), box.top(), box.width(), box.height(), matrix, true)
        //(3)save
        val pixelsBuf = IntArray(size * size)
        croped.getPixels(pixelsBuf, 0, croped.width, 0, 0, croped.width, croped.height)
        val imageMean = 127.5f
        val imageStd = 128f
        for (i in pixelsBuf.indices) {
            val `val` = pixelsBuf[i]
            data[i * 3 + 0] = ((`val` shr 16 and 0xFF) - imageMean) / imageStd
            data[i * 3 + 1] = ((`val` shr 8 and 0xFF) - imageMean) / imageStd
            data[i * 3 + 2] = ((`val` and 0xFF) - imageMean) / imageStd
        }
    }

    /*
     * RNET跑神经网络，将score和bias写入boxes
     */
    private fun refineNetForward(RNetIn: FloatArray, boxes: Vector<Box>) {
        val num = RNetIn.size / 24 / 24 / 3
        //feed & run
        inferenceInterface.feed(rNetInName, RNetIn, num.toLong(), 24, 24, 3)
        inferenceInterface.run(rNetOutName, false)
        //fetch
        val rNetP = FloatArray(num * 2)
        val rNetB = FloatArray(num * 4)
        inferenceInterface.fetch(rNetOutName[0], rNetP)
        inferenceInterface.fetch(rNetOutName[1], rNetB)
        //转换
        for (i in 0 until num) {
            boxes[i].score = rNetP[i * 2 + 1]
            for (j in 0..3)
                boxes[i].bbr[j] = rNetB[i * 4 + j]
        }
    }

    //Refine Net
    private fun refineNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        //refineNet Input Init
        val num = boxes.size
        val rNetIn = FloatArray(num * 24 * 24 * 3)
        val curCrop = FloatArray(24 * 24 * 3)
        var rNetInIdx = 0
        for (i in 0 until num) {
            cropAndResize(bitmap, boxes.get(i), 24, curCrop)
            PicUtils.flipDiag(curCrop, 24, 24, 3)
            //Log.i(TAG,"[*]Pixels values:"+curCrop[0]+" "+curCrop[1]);
            for (j in curCrop.indices) rNetIn[rNetInIdx++] = curCrop[j]
        }
        //Run refineNet
        refineNetForward(rNetIn, boxes)
        //RNetThreshold
        for (i in 0 until num)
            if (boxes[i].score < rNetThreshold)
                boxes[i].deleted = true
        //Nms
        nms(boxes, 0.7f, "Union")
        boundingBoxRegression(boxes)
        return PicUtils.updateBoxes(boxes)
    }

    /*
     * outputNet跑神经网络，将score和bias写入boxes
     */
    private fun outputNetForward(outputNetIn: FloatArray, boxes: Vector<Box>) {
        val num = outputNetIn.size / 48 / 48 / 3
        //feed & run
        inferenceInterface.feed(outputNetInName, outputNetIn, num.toLong(), 48, 48, 3)
        inferenceInterface.run(outputNetOutName, false)
        //fetch
        val outputNetP = FloatArray(num * 2) //prob
        val outputNetB = FloatArray(num * 4) //bias
        val outputNetL = FloatArray(num * 10) //landmark
        inferenceInterface.fetch(outputNetOutName[0], outputNetP)
        inferenceInterface.fetch(outputNetOutName[1], outputNetB)
        inferenceInterface.fetch(outputNetOutName[2], outputNetL)
        //转换
        for (i in 0 until num) {
            //prob
            boxes[i].score = outputNetP[i * 2 + 1]
            //bias
            for (j in 0..3)
                boxes[i].bbr[j] = outputNetB[i * 4 + j]

            //landmark
            for (j in 0..4) {
                val x = boxes[i].left() + (outputNetL[i * 10 + j] * boxes[i].width()).toInt()
                val y = boxes[i].top() + (outputNetL[i * 10 + j + 5] * boxes[i].height()).toInt()
                boxes[i].landmark[j] = Point(x, y)
            }
        }
    }

    //outputNet
    private fun outputNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        //outputNet Input Init
        val num = boxes.size
        val outputNetIn = FloatArray(num * 48 * 48 * 3)
        val curCrop = FloatArray(48 * 48 * 3)
        var outputNetInIdx = 0
        for (i in 0 until num) {
            cropAndResize(bitmap, boxes[i], 48, curCrop)
            PicUtils.flipDiag(curCrop, 48, 48, 3)
            for (j in curCrop.indices) outputNetIn[outputNetInIdx++] = curCrop[j]
        }
        //Run outputNet
        outputNetForward(outputNetIn, boxes)
        //outputNetThreshold
        for (i in 0 until num)
            if (boxes[i].score < outputNetThreshold)
                boxes[i].deleted = true
        boundingBoxRegression(boxes)
        //Nms
        nms(boxes, 0.7f, "Min")
        return PicUtils.updateBoxes(boxes)
    }

    private fun squareLimit(boxes: Vector<Box>, w: Int, h: Int) {
        //square
        for (i in 0 until boxes.size) {
            boxes[i].toSquareShape()
            boxes[i].limitSquare(w, h)
        }
    }

    /*
     * 参数：
     *   bitmap:要处理的图片
     *   minFaceSize:最小的人脸像素值.(此值越大，检测越快)
     * 返回：
     *   人脸框
     */
    fun detectFaces(bitmap: Bitmap, minFaceSize: Int): Vector<Box> {
        val tStart = System.currentTimeMillis()
        //【1】proposalNet generate candidate boxes
        var boxes = proposalNet(bitmap, minFaceSize)
        squareLimit(boxes, bitmap.width, bitmap.height)
        //【2】refineNet
        boxes = refineNet(bitmap, boxes)
        squareLimit(boxes, bitmap.width, bitmap.height)
        //【3】outputNet
        boxes = outputNet(bitmap, boxes)
        //return
        lastProcessTime = System.currentTimeMillis() - tStart
        Log.i(TAG, "[*]Mtcnn Detection Time:$lastProcessTime")
        return boxes
    }

    fun cutFace(bitmap: Bitmap? , boxes: Vector<Box>): List<Bitmap> {
        if(bitmap == null){
            throw IllegalArgumentException("no images!")
        }
        val findFaceBitmap = PicUtils.copyBitmap(bitmap)
        val faces = mutableListOf<Bitmap>()
        boxes.forEach{
            PicUtils.drawRect(findFaceBitmap, it.transform2Rect())
            PicUtils.drawPoints(findFaceBitmap, it.landmark)
            PicUtils.rectExtend(findFaceBitmap , it.transform2Rect() , 20)
            faces.add(Bitmap.createScaledBitmap(PicUtils.crop(findFaceBitmap , boxes[0].transform2Rect()),160,160,true))
        }
        return faces
    }
}

