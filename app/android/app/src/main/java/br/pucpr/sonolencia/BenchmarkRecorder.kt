package br.pucpr.sonolencia

import android.content.Context
import android.os.Build
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Grava o benchmark de 60 s: um CSV com uma linha por quadro e um JSON de resumo, em
 * `getExternalFilesDir("benchmarks")` (pasta privada do app, acessível via
 * Android/data/br.pucpr.sonolencia/files/benchmarks e pelo botão Compartilhar).
 *
 * Privacidade: só métricas numéricas são gravadas; nenhum quadro/imagem é salvo.
 */
class BenchmarkRecorder(context: Context, val durationMs: Long = 60_000L) {
    val dir: File = dirFor(context)
    private val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val csvFile = File(dir, "bench_$stamp.csv")
    val jsonFile = File(dir, "bench_$stamp.json")
    private val writer = BufferedWriter(FileWriter(csvFile))
    private var startUptime = -1L
    private var rows = 0
    private var rowsWithFace = 0
    private val latLandmarker = LatencyStats(100_000)
    private val latCnn = LatencyStats(100_000)
    private val latEngine = LatencyStats(100_000)
    private val latTotal = LatencyStats(100_000)
    @Volatile var finished = false
        private set

    init {
        writer.write(HEADER)
        writer.newLine()
    }

    val elapsedMs: Long get() = if (startUptime < 0) 0L else android.os.SystemClock.uptimeMillis() - startUptime

    @Synchronized
    fun addRow(
        uptimeMs: Long, face: Boolean, landmarkerMs: Float, cnnMs: Float, engineMs: Float, totalMs: Float,
        out: EngineOutput?, pOpenL: Float?, pOpenR: Float?,
    ): Boolean {
        if (finished) return false
        if (startUptime < 0) startUptime = uptimeMs
        rows++
        if (face) rowsWithFace++
        latLandmarker.add(landmarkerMs)
        latTotal.add(totalMs)
        if (face) {
            latCnn.add(cnnMs)
            latEngine.add(engineMs)
        }
        val p = out?.probabilities
        val line = listOf(
            rows, uptimeMs - startUptime, if (face) 1 else 0,
            f(landmarkerMs), f(cnnMs), f(engineMs), f(totalMs),
            f(out?.ear), f(out?.mar), f(out?.perclos), f(pOpenL), f(pOpenR),
            if (out?.closed == true) 1 else 0, out?.blinkCount ?: 0, out?.classIndex ?: -1,
            f(p?.getOrNull(0)), f(p?.getOrNull(1)), f(p?.getOrNull(2)),
            out?.phase?.name ?: "", out?.alerts?.joinToString("|") { it.name } ?: "",
        ).joinToString(",")
        writer.write(line)
        writer.newLine()
        if (rows % 30 == 0) writer.flush()
        return uptimeMs - startUptime >= durationMs
    }

    private fun f(v: Float?): String = if (v == null || v.isNaN()) "" else String.format(Locale.US, "%.4f", v)

    /** Fecha o CSV e grava o resumo JSON. */
    @Synchronized
    fun finish(
        delegate: String?, modelSizes: Map<String, Long>, eyeModelAsset: String?, temporalAsset: String?,
    ): File {
        if (finished) return jsonFile
        finished = true
        writer.flush()
        writer.close()
        val durationS = elapsedMs / 1000.0
        val summary = JSONObject().apply {
            put("app", "br.pucpr.sonolencia")
            put("created_at", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(Date()))
            put("device", JSONObject().apply {
                put("manufacturer", Build.MANUFACTURER)
                put("model", Build.MODEL)
                put("device", Build.DEVICE)
                put("hardware", Build.HARDWARE)
                put("soc_model", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "n/d")
                put("soc_manufacturer", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MANUFACTURER else "n/d")
            })
            put("android", JSONObject().apply {
                put("release", Build.VERSION.RELEASE)
                put("sdk_int", Build.VERSION.SDK_INT)
            })
            put("landmarker_delegate", delegate ?: "n/d")
            put("eye_model", eyeModelAsset ?: "ausente")
            put("temporal_model", temporalAsset ?: "ausente")
            put("model_sizes_bytes", JSONObject(modelSizes))
            put("duration_s", durationS)
            put("frames", rows)
            put("frames_with_face", rowsWithFace)
            put("fps_mean", if (durationS > 0) rows / durationS else 0.0)
            put("latency_ms", JSONObject().apply {
                put("landmarker", statsJson(latLandmarker))
                put("cnn", statsJson(latCnn))
                put("engine", statsJson(latEngine))
                put("total", statsJson(latTotal))
            })
            put("csv", csvFile.name)
        }
        jsonFile.writeText(summary.toString(2))
        return jsonFile
    }

    private fun statsJson(s: LatencyStats) = JSONObject().apply {
        put("n", s.count)
        put("mean", s.mean().toDouble())
        put("p50", s.p50().toDouble())
        put("p95", s.p95().toDouble())
    }

    companion object {
        /** Pasta dos benchmarks: Android/data/br.pucpr.sonolencia/files/benchmarks. */
        fun dirFor(context: Context): File =
            File(context.getExternalFilesDir(null), "benchmarks").apply { mkdirs() }

        const val HEADER = "frame,t_ms,face,landmarker_ms,cnn_ms,engine_ms,total_ms,ear,mar,perclos," +
            "p_open_l,p_open_r,closed,blink_count,class,p_alert,p_low_vigilant,p_drowsy,phase,alerts"
    }
}
