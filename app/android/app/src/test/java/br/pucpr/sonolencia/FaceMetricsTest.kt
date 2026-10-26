package br.pucpr.sonolencia

import org.junit.Assert.assertEquals
import org.junit.Test

class FaceMetricsTest {
    private fun p(x: Float, y: Float) = FaceMetrics.Point(x, y)

    @Test
    fun earOlhoAbertoSintetico() {
        // p1=(0,0) p4=(10,0): largura 10; pálpebras a +-1.5 e +-1.0 => (3+2)/(2*10) = 0.25
        val pts = listOf(p(0f, 0f), p(3f, -1.5f), p(7f, -1f), p(10f, 0f), p(7f, 1f), p(3f, 1.5f))
        assertEquals(0.25f, FaceMetrics.eyeAspectRatio(pts), 1e-6f)
    }

    @Test
    fun earOlhoFechadoEhZero() {
        val pts = listOf(p(0f, 0f), p(3f, 0f), p(7f, 0f), p(10f, 0f), p(7f, 0f), p(3f, 0f))
        assertEquals(0f, FaceMetrics.eyeAspectRatio(pts), 1e-6f)
    }

    @Test
    fun earLarguraZeroNaoDivide() {
        val pts = List(6) { p(1f, 1f) }
        assertEquals(0f, FaceMetrics.eyeAspectRatio(pts), 1e-6f)
    }

    @Test
    fun marSintetico() {
        assertEquals(0.5f, FaceMetrics.mouthAspectRatio(p(0f, 0f), p(20f, 0f), p(10f, -5f), p(10f, 5f)), 1e-6f)
    }

    @Test
    fun indicesDoPrototipo() {
        assertEquals(listOf(362, 385, 387, 263, 373, 380), FaceMetrics.LEFT_EYE.toList())
        assertEquals(listOf(33, 160, 158, 133, 153, 144), FaceMetrics.RIGHT_EYE.toList())
        assertEquals(listOf(61, 291), FaceMetrics.MOUTH_CORNERS.toList())
        assertEquals(listOf(13, 14), FaceMetrics.MOUTH_VERTICAL.toList())
    }

    @Test
    fun landmarksEyeBoxUsaCantosP1P4() {
        val xy = FloatArray(FaceMetrics.NUM_LANDMARKS * 2)
        // Olho esquerdo: p1 em (100,50), p4 em (120,50), demais no meio.
        val idx = FaceMetrics.LEFT_EYE
        val coords = listOf(100f to 50f, 105f to 48f, 115f to 48f, 120f to 50f, 115f to 52f, 105f to 52f)
        for (k in idx.indices) {
            xy[2 * idx[k]] = coords[k].first
            xy[2 * idx[k] + 1] = coords[k].second
        }
        val lm = FaceMetrics.Landmarks(xy)
        val box = lm.eyeBox(idx)
        assertEquals(110f, box.cx, 1e-4f)
        assertEquals(50f, box.cy, 1e-4f)
        assertEquals(1.6f * 20f, box.side, 1e-4f)
        // EAR = (4 + 4) / (2 * 20) = 0.2
        assertEquals(0.2f, lm.leftEar(), 1e-6f)
    }
}
