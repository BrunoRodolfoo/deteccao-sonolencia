package br.pucpr.sonolencia

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Testes de integração da lógica pura, com sinal de EAR sintético a 30 fps. */
class DrowsinessEngineTest {
    private val frameMs = 33L

    private class FakeClassifier(var probs: FloatArray) : TemporalClassifier {
        var calls = 0
        var lastWindow: Array<FloatArray>? = null
        override fun predict(window: Array<FloatArray>): FloatArray {
            calls++
            lastWindow = window
            return probs
        }
    }

    private val config = EngineConfig(
        thresholdCalibFrames = 30, calibBlinks = 5, minCalibBlinks = 2,
        perclosMinCoverageMs = 500L,
    )

    /** Simulador: mantém o relógio e alimenta o motor. */
    private class Sim(val engine: DrowsinessEngine, val frameMs: Long) {
        var t = 0L
        var last: EngineOutput? = null
        /** Nº de quadros em que o motor reportou uma piscada concluída (newBlink != null). */
        var blinkEvents = 0
        fun frames(n: Int, ear: Float, mar: Float = 0.2f): EngineOutput {
            repeat(n) {
                t += frameMs
                val o = engine.processFrame(t, ear, ear, mar)
                if (o.newBlink != null) blinkEvents++
                last = o
            }
            return last!!
        }
        fun blink(): EngineOutput {
            frames(1, 0.15f); frames(1, 0.05f); frames(1, 0.15f)
            return frames(20, 0.30f)
        }
    }

    @Test
    fun calibracaoCompletaEClassifica() {
        val clf = FakeClassifier(floatArrayOf(0.8f, 0.1f, 0.1f))
        val engine = DrowsinessEngine(config, clf)
        val sim = Sim(engine, frameMs)

        assertEquals(EnginePhase.IDLE, engine.phase)
        engine.start()
        assertEquals(EnginePhase.THRESHOLD_CALIB, engine.phase)

        val out0 = sim.frames(30, 0.30f)
        assertEquals(EnginePhase.BLINK_CALIB, out0.phase)
        assertEquals(0.75f * 0.30f, engine.threshold!!, 1e-6f)

        repeat(5) { sim.blink() }
        assertEquals(EnginePhase.RUNNING, engine.phase)
        assertEquals(5L, engine.blinkTotal)
        assertNotNull(engine.normalizerOrNull)
        assertEquals(1, clf.calls)   // inferência ao fim da calibração

        // Janela: 25 linhas de zeros à esquerda + 5 piscadas normalizadas
        val w = clf.lastWindow!!
        assertEquals(30, w.size)
        for (i in 0 until 25) assertTrue(w[i].all { it == 0f })
        val out = sim.blink()
        assertEquals(2, clf.calls)
        assertEquals(0, out.classIndex)
        assertTrue(out.alerts.isEmpty())
        assertEquals(6, sim.blinkEvents)
    }

    @Test
    fun alertaSonolentoAposTresJanelasSustentadas() {
        val clf = FakeClassifier(floatArrayOf(0.1f, 0.1f, 0.8f))
        val engine = DrowsinessEngine(config, clf)
        val sim = Sim(engine, frameMs)
        engine.start()
        sim.frames(30, 0.30f)
        repeat(5) { sim.blink() }          // fim da calibração => janela 1 (sonolento)
        var out = sim.blink()               // janela 2
        assertFalse(AlertType.DROWSY in out.alerts)
        out = sim.blink()                   // janela 3
        assertTrue(AlertType.DROWSY in out.alerts)
        assertEquals(2, out.classIndex)

        // Ao voltar para "alerta", a suavização exponencial demora algumas janelas.
        clf.probs = floatArrayOf(0.9f, 0.05f, 0.05f)
        out = sim.blink()
        assertTrue(out.probabilities!![2] < 0.8f)
    }

    @Test
    fun alertaOlhosFechadosApos1s() {
        val engine = DrowsinessEngine(config, null)
        val sim = Sim(engine, frameMs)
        engine.start()
        sim.frames(30, 0.30f)
        var out = sim.frames(20, 0.05f)     // ~0.66 s
        assertFalse(AlertType.EYES_CLOSED in out.alerts)
        assertTrue(out.closed)
        out = sim.frames(15, 0.05f)          // ~1.15 s
        assertTrue(AlertType.EYES_CLOSED in out.alerts)
        assertTrue(out.closedMs >= 1000L)
        out = sim.frames(15, 0.05f)          // ~1.65 s: acima do máximo de 1,5 s de uma piscada
        assertTrue(AlertType.EYES_CLOSED in out.alerts)
        out = sim.frames(1, 0.30f)
        assertEquals(0L, engine.blinkTotal)  // fechamento longo não conta como piscada
        assertFalse(AlertType.EYES_CLOSED in out.alerts)
        assertFalse(out.closed)
    }

    @Test
    fun alertaPerclos() {
        val engine = DrowsinessEngine(config, null)
        val sim = Sim(engine, frameMs)
        engine.start()
        sim.frames(30, 0.30f)
        // 30% do tempo fechado em fechamentos curtos (piscadas) => PERCLOS > 0.15
        var out: EngineOutput? = null
        repeat(20) {
            sim.frames(3, 0.05f)
            out = sim.frames(7, 0.30f)
        }
        assertTrue(out!!.perclos > 0.15f)
        assertTrue(AlertType.PERCLOS in out!!.alerts)
    }

    @Test
    fun semClassificadorFuncionaDegradado() {
        val engine = DrowsinessEngine(config, null)
        val sim = Sim(engine, frameMs)
        engine.start()
        sim.frames(30, 0.30f)
        repeat(6) { sim.blink() }
        val out = sim.blink()
        assertEquals(EnginePhase.RUNNING, out.phase)
        assertNull(out.probabilities)
        assertEquals(-1, out.classIndex)
        assertEquals(7L, out.blinkCount)
    }

    @Test
    fun semRostoNaoContaQuadro() {
        val engine = DrowsinessEngine(config, null)
        engine.start()
        val out = engine.processNoFace(100L)
        assertFalse(out.faceDetected)
        assertEquals(EnginePhase.THRESHOLD_CALIB, out.phase)
        assertEquals(0, out.calibFramesDone)
    }

    @Test
    fun recalibrarPiscadasMantemLimiar() {
        val clf = FakeClassifier(floatArrayOf(0.8f, 0.1f, 0.1f))
        val engine = DrowsinessEngine(config, clf)
        val sim = Sim(engine, frameMs)
        engine.start()
        sim.frames(30, 0.30f)
        repeat(5) { sim.blink() }
        assertEquals(EnginePhase.RUNNING, engine.phase)
        val thr = engine.threshold
        engine.recalibrateBlinks(sim.t)
        assertEquals(EnginePhase.BLINK_CALIB, engine.phase)
        assertEquals(thr, engine.threshold)
        assertNull(engine.normalizerOrNull)
        repeat(5) { sim.blink() }
        assertEquals(EnginePhase.RUNNING, engine.phase)
    }

    @Test
    fun medianaEArgmax() {
        assertEquals(2f, DrowsinessEngine.median(listOf(3f, 1f, 2f)), 0f)
        assertEquals(2.5f, DrowsinessEngine.median(listOf(4f, 1f, 2f, 3f)), 0f)
        assertEquals(1, DrowsinessEngine.argmax(floatArrayOf(0.2f, 0.7f, 0.1f)))
    }
}
