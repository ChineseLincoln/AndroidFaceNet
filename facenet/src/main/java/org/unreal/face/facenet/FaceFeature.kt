package org.unreal.face.facenet

class FaceFeature internal constructor() {
    val feature: FloatArray

    init {
        feature = FloatArray(DIMS)
    }

    //比较当前特征和另一个特征之间的相似度
    fun compare(ff: FaceFeature): Double {
        var dist = 0.0
        for (i in 0 until DIMS)
            dist += ((feature[i] - ff.feature[i]) * (feature[i] - ff.feature[i])).toDouble()
        dist = Math.sqrt(dist)
        return dist
    }

    companion object {
        const val DIMS = 512
    }
}