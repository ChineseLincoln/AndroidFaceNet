package org.unreal.face.recognition.activity

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import kotlinx.android.synthetic.main.activity_main.*
import org.unreal.face.facenet.BitmapUtils
import org.unreal.face.facenet.FaceNet
import org.unreal.face.mtcnn.MTCNN
import org.unreal.face.mtcnn.PicUtils
import org.unreal.face.mtcnn.PicUtils.copyBitmap
import org.unreal.face.mtcnn.PicUtils.drawPoints
import org.unreal.face.mtcnn.PicUtils.drawRect
import org.unreal.face.recognition.R


class MainActivity : AppCompatActivity() {

    private lateinit var mtcnn: MTCNN
    private lateinit var faceNet: FaceNet

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initFaceEngine()
        setContentView(R.layout.activity_main)

        val startTime = System.currentTimeMillis()
        val source = PicUtils.getBitmapFromAssets(assets, "1.jpg")
        val dist = PicUtils.getBitmapFromAssets(assets, "4.jpg")
        imageView2.setImageBitmap(dist)

        val findFaceSource = mtcnn.cutFace(source,mtcnn.detectFaces(source , 40 ))
        imageView.setImageBitmap(findFaceSource[0])

        val findFaceDist =  mtcnn.cutFace(dist,mtcnn.detectFaces(dist , 40 ))
        imageView2.setImageBitmap(findFaceDist[0])


        val compare = faceNet.recognizeImage(findFaceSource[0]).compare(faceNet.recognizeImage(findFaceDist[0]))
        val endTime = System.currentTimeMillis()
        Log.e("MainActivity","compare is --->$compare")
        Log.e("MainActivity","pass is --->${compare < 0.8}")
        Log.e("MainActivity","run time is ${endTime - startTime} ms")
    }

    private fun initFaceEngine() {
        mtcnn = MTCNN(assets)
        faceNet = FaceNet(assets)
    }


}
