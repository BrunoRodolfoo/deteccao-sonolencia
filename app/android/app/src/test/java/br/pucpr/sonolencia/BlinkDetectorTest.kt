package br.pucpr.sonolencia

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BlinkDetectorTest {
    private val fps = 30
    private val frameMs = 1000L / fps

    /** Alimenta o detector com uma série de EAR a 30 fps; devolve as piscadas detectadas. */
    private fun run(det: BlinkDetector, ears: List<Float>): List<Blink> {
        val out = ArrayList<Blink>()
        ears.forEachIndexed { i, e -> det.update(i.toLong(), i * frameMs, e)?.let { out.add(it) } }
        return out
    }

    @Test
    fun detectaPiscadaSimplesComStartBottomEnd() {
        val det = BlinkDetector().apply { threshold = 0.225f }
        val ears = List(10) { 0.30f } + listOf(0.20f, 0.05f, 0.10f, 0.21f) + List(10) { 0.30f }
        val blinks = run(det, ears)
        assertEquals(1, blinks.size)
        val b = blinks[0]
        assertEquals(10L, b.startFrame)
        assertEquals(11L, b.bottomFrame)
        assertEquals(13L, b.endFrame)
        assertEquals(4L, b.durationFrames)
        assertEquals(0.20f, b.earStart, 1e-6f)
        assertEquals(0.05f, b.earBottom, 1e-6f)
        assertEquals(0.21f, b.earEnd, 1e-6f)
    }

    @Test
    fun descartaRuidoDeUmQuadro() {
        val det = BlinkDetector(minFrames = 2).apply { threshold = 0.225f }
        val ears = List(5) { 0.30f } + listOf(0.10f) + List(5) { 0.30f }
        assertTrue(run(det, ears).isEmpty())
    }

    @Test
    fun descartaFechamentoLongoMasMarcaFechado() {
        val det = BlinkDetector(maxDurationMs = 1500L).apply { threshold = 0.225f }
        val ears = List(5) { 0.30f } + List(60) { 0.05f } + List(5) { 0.30f }   // 2 s fechado
        var closedFrames = 0
        var blinks = 0
        ears.forEachIndexed { i, e ->
            if (det.update(i.toLong(), i * frameMs, e) != null) blinks++
            if (det.isClosed) closedFrames++
        }
        assertEquals(0, blinks)
        assertEquals(60, closedFrames)
    }

    @Test
    fun semLimiarNaoDetecta() {
        val det = BlinkDetector()
        assertNull(det.update(0, 0, 0.0f))
        assertNull(det.update(1, 33, 0.3f))
    }

    @Test
    fun contaTresPiscadas() {
        val det = BlinkDetector().apply { threshold = 0.225f }
        val blink = listOf(0.15f, 0.05f, 0.15f)
        val open = List(20) { 0.30f }
        val ears = open + blink + open + blink + open + blink + open
        assertEquals(3, run(det, ears).size)
    }

    @Test
    fun caracteristicasSeguemAsFormulasDoArtigo() {
        val b = Blink(
            startFrame = 100, bottomFrame = 102, endFrame = 105,
            earStart = 0.20f, earBottom = 0.04f, earEnd = 0.22f,
            startTimeMs = 0, endTimeMs = 166,
        )
        // 7ª piscada; 300 quadros até end
        val f = BlinkFeatures.fromBlink(b, blinkCountUpToThis = 7, framesUntilEnd = 300)
        assertEquals(6f, f.duration, 1e-6f)                                   // 105-100+1
        assertEquals((0.20f - 2 * 0.04f + 0.22f) / 2f, f.amplitude, 1e-6f)   // 0.17
        assertEquals((0.22f - 0.04f) / 3f, f.velocity, 1e-6f)                // (end-bottom)=3
        assertEquals(100f * 7f / 300f, f.frequency, 1e-6f)
        // Ordem do vetor: Freq, Amp, Dur, Vel
        val arr = f.toArray()
        assertEquals(f.frequency, arr[0], 0f)
        assertEquals(f.amplitude, arr[1], 0f)
        assertEquals(f.duration, arr[2], 0f)
        assertEquals(f.velocity, arr[3], 0f)
    }

    @Test
    fun velocidadeComEndIgualBottomUsaPisoDeUmQuadro() {
        val b = Blink(0, 3, 3, 0.2f, 0.05f, 0.05f, 0, 100)
        val f = BlinkFeatures.fromBlink(b, 1, 10)
        assertEquals(0f, f.velocity, 1e-6f)
        assertNotNull(f)
    }

    @Test
    fun normalizadorZScore() {
        val samples = listOf(
            floatArrayOf(1f, 10f, 5f, 0.1f),
            floatArrayOf(3f, 10f, 7f, 0.3f),
        )
        val n = FeatureNormalizer.fit(samples)
        assertEquals(2f, n.mean[0], 1e-6f)
        assertEquals(1f, n.std[0], 1e-6f)
        assertEquals(FeatureNormalizer.STD_FLOOR, n.std[1], 0f)   // desvio zero -> piso
        val z = n.normalize(floatArrayOf(4f, 10f, 6f, 0.2f))
        assertEquals(2f, z[0], 1e-6f)
        assertEquals(0f, z[1], 1e-6f)
        assertEquals(0f, z[2], 1e-6f)
        assertEquals(0f, z[3], 1e-5f)
    }
}
