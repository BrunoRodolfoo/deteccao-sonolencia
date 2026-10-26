package br.pucpr.sonolencia

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import br.pucpr.sonolencia.databinding.ActivityMainBinding
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.File
import java.util.ArrayDeque
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Tela única: pré-visualização da câmera frontal + HUD + botões.
 *
 * Pipeline por quadro (thread do ImageAnalysis -> thread de callback do MediaPipe):
 *   CameraX (640x480 RGBA) -> Bitmap girado/espelhado -> FaceLandmarker LIVE_STREAM
 *   -> EAR/MAR (FaceMetrics) -> recorte dos olhos -> CNN de estado do olho (TFLite)
 *   -> DrowsinessEngine (piscadas, PERCLOS, classificador temporal TFLite) -> HUD/alertas.
 *
 * Privacidade: os quadros ficam apenas em memória e são descartados após o processamento;
 * o benchmark grava somente números (CSV/JSON).
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private val mainHandler = Handler(Looper.getMainLooper())

    private var landmarker: FaceLandmarkerHelper? = null
    private var temporalModel: BlinkTemporalModel? = null
    private var eyeModel: EyeStateModel? = null
    private var eyeModelAsset: String? = null
    private var engine: DrowsinessEngine = DrowsinessEngine()
    private val engineLock = Any()
    private lateinit var alerter: Alerter

    private class PendingFrame(val bitmap: Bitmap, val submitUptimeMs: Long)

    private val pending = ConcurrentHashMap<Long, PendingFrame>()
    private var lastTimestampMs = 0L

    @Volatile private var running = false
    @Volatile private var benchmark: BenchmarkRecorder? = null
    private var lastBenchmarkFiles: List<File> = emptyList()

    private val latLandmarker = LatencyStats(300)
    private val latCnn = LatencyStats(300)
    private val latTotal = LatencyStats(300)
    private val frameTimes = ArrayDeque<Long>()   // para FPS (quadros processados no último 1 s)

    @Volatile private var lastOutput: EngineOutput? = null
    @Volatile private var lastOpenL: Float? = null
    @Volatile private var lastOpenR: Float? = null
    private var lastUiUpdateMs = 0L

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, R.string.perm_camera_denied, Toast.LENGTH_LONG).show()
            binding.txtStatus.setText(R.string.perm_camera_denied)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()
        alerter = Alerter(this)

        loadModels()

        binding.btnStart.setOnClickListener { if (running) stopEngine() else startEngine() }
        binding.btnCalibrate.setOnClickListener {
            synchronized(engineLock) { engine.recalibrateBlinks(SystemClock.uptimeMillis()) }
        }
        binding.btnBenchmark.setOnClickListener { startBenchmark() }
        binding.btnShare.setOnClickListener { shareBenchmark() }
        // Toque no texto dos modelos alterna a CNN entre int8 e fp32 (se ambos existirem).
        binding.txtModels.setOnClickListener { toggleEyeModel() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    // ------------------------------------------------------------------ modelos

    private fun loadModels() {
        val lm = FaceLandmarkerHelper(this, ::onLandmarkerResult, ::onLandmarkerError)
        val err = lm.setup()
        if (err != null) {
            Log.e(TAG, "FaceLandmarker: $err")
            landmarker = null
            binding.txtStatus.setText(R.string.landmarker_missing)
        } else {
            landmarker = lm
        }

        temporalModel = try {
            if (TfliteLoader.assetExists(this, BlinkTemporalModel.ASSET)) BlinkTemporalModel(this) else null
        } catch (e: Exception) {
            Log.e(TAG, "Falha ao carregar ${BlinkTemporalModel.ASSET}", e); null
        }
        eyeModelAsset = when {
            TfliteLoader.assetExists(this, EyeStateModel.ASSET_INT8) -> EyeStateModel.ASSET_INT8
            TfliteLoader.assetExists(this, EyeStateModel.ASSET_FP32) -> EyeStateModel.ASSET_FP32
            else -> null
        }
        eyeModel = loadEyeModel(eyeModelAsset)
        engine = DrowsinessEngine(EngineConfig(), temporalModel)
        updateModelsText()
        if (temporalModel == null || eyeModel == null) {
            Toast.makeText(this, R.string.degraded_mode, Toast.LENGTH_LONG).show()
        }
    }

    private fun loadEyeModel(asset: String?): EyeStateModel? {
        if (asset == null) return null
        return try {
            EyeStateModel(this, asset)
        } catch (e: Exception) {
            Log.e(TAG, "Falha ao carregar $asset", e); null
        }
    }

    private fun toggleEyeModel() {
        val other = when (eyeModelAsset) {
            EyeStateModel.ASSET_INT8 -> EyeStateModel.ASSET_FP32
            EyeStateModel.ASSET_FP32 -> EyeStateModel.ASSET_INT8
            else -> return
        }
        if (!TfliteLoader.assetExists(this, other)) return
        val newModel = loadEyeModel(other) ?: return
        synchronized(engineLock) {
            eyeModel?.close()
            eyeModel = newModel
            eyeModelAsset = other
            latCnn.reset()
        }
        updateModelsText()
    }

    private fun updateModelsText() {
        val lmName = landmarker?.let { "${FaceLandmarkerHelper.MODEL_ASSET} (${it.delegateName})" }
            ?: getString(R.string.model_missing)
        val tName = if (temporalModel != null) BlinkTemporalModel.ASSET else getString(R.string.model_missing)
        val eName = eyeModelAsset ?: getString(R.string.model_missing)
        binding.txtModels.text = getString(R.string.models_ok, lmName, tName, eName)
    }

    private fun modelSizes(): Map<String, Long> {
        val names = listOf(
            FaceLandmarkerHelper.MODEL_ASSET, BlinkTemporalModel.ASSET,
            EyeStateModel.ASSET_INT8, EyeStateModel.ASSET_FP32,
        )
        return names.associateWith { TfliteLoader.assetSizeBytes(this, it) }.filterValues { it >= 0 }
    }

    // ------------------------------------------------------------------ câmera

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build()
                preview.setSurfaceProvider(binding.previewView.surfaceProvider)

                val selector = ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                    )
                    .build()
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(selector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(cameraExecutor) { proxy -> analyzeFrame(proxy) }

                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "Erro ao iniciar a câmera", e)
                Toast.makeText(this, getString(R.string.camera_error, e.message ?: ""), Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Thread do ImageAnalysis: converte o quadro e envia ao FaceLandmarker. */
    private fun analyzeFrame(proxy: ImageProxy) {
        try {
            val lm = landmarker
            if (!running || lm == null || !lm.isReady) return
            // Controle de fluxo: no máximo 2 quadros em voo; descarta quadros pendentes antigos.
            val now = SystemClock.uptimeMillis()
            pending.entries.removeIf { now - it.value.submitUptimeMs > 2000L }
            if (pending.size >= 2) return

            val bitmap = proxy.toBitmap()   // RGBA_8888 -> ARGB_8888 (cópia em memória)
            val rotation = proxy.imageInfo.rotationDegrees
            val matrix = Matrix().apply {
                postRotate(rotation.toFloat())
                // Espelha horizontalmente para coincidir com a pré-visualização da câmera frontal.
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
            val upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            // Timestamp estritamente crescente exigido pelo modo LIVE_STREAM.
            var ts = SystemClock.uptimeMillis()
            if (ts <= lastTimestampMs) ts = lastTimestampMs + 1
            lastTimestampMs = ts
            pending[ts] = PendingFrame(upright, ts)
            lm.detectAsync(upright, ts)
        } catch (e: Exception) {
            Log.e(TAG, "Erro no analisador", e)
        } finally {
            proxy.close()
        }
    }

    // ------------------------------------------------------------------ resultado do landmarker

    private fun onLandmarkerError(e: RuntimeException) {
        Log.e(TAG, "Erro do FaceLandmarker", e)
    }

    /** Thread interna do MediaPipe. */
    private fun onLandmarkerResult(result: FaceLandmarkerResult, input: MPImage) {
        val ts = result.timestampMs()
        val frame = pending.remove(ts) ?: return
        val now = SystemClock.uptimeMillis()
        val landmarkerMs = (now - frame.submitUptimeMs).toFloat()
        val bench = benchmark

        val faces = result.faceLandmarks()
        if (faces.isEmpty()) {
            val out = synchronized(engineLock) { engine.processNoFace(now) }
            lastOutput = out
            lastOpenL = null; lastOpenR = null
            val total = (SystemClock.uptimeMillis() - frame.submitUptimeMs).toFloat()
            latLandmarker.add(landmarkerMs); latTotal.add(total)
            tickFps(now)
            bench?.let { if (it.addRow(now, false, landmarkerMs, 0f, 0f, total, out, null, null)) finishBenchmark(it) }
            scheduleUi()
            return
        }

        // Landmarks normalizados -> pixels do bitmap analisado.
        val lmList = faces[0]
        val w = frame.bitmap.width.toFloat()
        val h = frame.bitmap.height.toFloat()
        val xy = FloatArray(lmList.size * 2)
        for (i in lmList.indices) {
            xy[2 * i] = lmList[i].x() * w
            xy[2 * i + 1] = lmList[i].y() * h
        }
        val lm = FaceMetrics.Landmarks(xy)
        val earL = lm.leftEar()
        val earR = lm.rightEar()
        val mar = lm.mar()

        // CNN de estado do olho (opcional).
        var cnnMs = 0f
        var pL: Float? = null
        var pR: Float? = null
        val t0 = System.nanoTime()
        synchronized(engineLock) {
            eyeModel?.let { model ->
                val boxL = lm.eyeBox(FaceMetrics.LEFT_EYE)
                val boxR = lm.eyeBox(FaceMetrics.RIGHT_EYE)
                pL = EyeCropper.cropGray(frame.bitmap, boxL, model.width, model.height)?.let { model.predictOpenProb(it) }
                pR = EyeCropper.cropGray(frame.bitmap, boxR, model.width, model.height)?.let { model.predictOpenProb(it) }
            }
        }
        if (eyeModel != null) cnnMs = (System.nanoTime() - t0) / 1e6f

        val t1 = System.nanoTime()
        val out = synchronized(engineLock) { engine.processFrame(now, earL, earR, mar) }
        val engineMs = (System.nanoTime() - t1) / 1e6f
        val total = (SystemClock.uptimeMillis() - frame.submitUptimeMs).toFloat()

        latLandmarker.add(landmarkerMs)
        latCnn.add(cnnMs)
        latTotal.add(total)
        tickFps(now)
        lastOutput = out
        lastOpenL = pL
        lastOpenR = pR

        if (out.alerts.isNotEmpty()) alerter.alert()

        bench?.let { if (it.addRow(now, true, landmarkerMs, cnnMs, engineMs, total, out, pL, pR)) finishBenchmark(it) }
        scheduleUi()
    }

    private fun tickFps(now: Long) {
        synchronized(frameTimes) {
            frameTimes.addLast(now)
            while (frameTimes.isNotEmpty() && now - frameTimes.first > 1000L) frameTimes.removeFirst()
        }
    }

    private fun currentFps(): Float = synchronized(frameTimes) {
        if (frameTimes.size < 2) {
            frameTimes.size.toFloat()
        } else {
            val span = (frameTimes.last - frameTimes.first).coerceAtLeast(1L)
            (frameTimes.size - 1) * 1000f / span
        }
    }

    // ------------------------------------------------------------------ HUD

    private fun scheduleUi() {
        val now = SystemClock.uptimeMillis()
        if (now - lastUiUpdateMs < 100L) return
        lastUiUpdateMs = now
        mainHandler.post { updateHud() }
    }

    private fun updateHud() {
        val out = lastOutput ?: return
        val cfg = engine.config
        binding.txtStatus.text = when {
            !running -> getString(R.string.status_idle)
            !out.faceDetected -> getString(R.string.status_no_face)
            out.phase == EnginePhase.THRESHOLD_CALIB ->
                getString(R.string.status_threshold_calib, out.calibFramesDone, cfg.thresholdCalibFrames)
            out.phase == EnginePhase.BLINK_CALIB ->
                getString(R.string.status_blink_calib, out.calibBlinksDone, cfg.calibBlinks, out.calibElapsedMs / 1000)
            else -> getString(R.string.status_running)
        }
        binding.txtFps.text = getString(R.string.hud_fps, currentFps())
        binding.txtLatency.text = getString(R.string.hud_latency, latLandmarker.last, latCnn.last, latTotal.last)
        binding.txtLatencyPct.text = getString(
            R.string.hud_latency_pct,
            latLandmarker.p50(), latLandmarker.p95(), latCnn.p50(), latCnn.p95(), latTotal.p50(), latTotal.p95(),
        )
        binding.txtMetrics.text = getString(R.string.hud_metrics, out.ear, out.threshold ?: 0f, out.mar, out.perclos)
        val pl = lastOpenL
        val pr = lastOpenR
        binding.txtEyes.text = if (pl != null && pr != null) getString(R.string.hud_eyes, pl, pr) else getString(R.string.hud_eyes_na)
        binding.txtBlinks.text = getString(R.string.hud_blinks, out.blinkCount, out.sequenceLength)
        val p = out.probabilities
        binding.txtClass.text = if (p != null && out.phase == EnginePhase.RUNNING) {
            val name = when (out.classIndex) {
                0 -> getString(R.string.class_alert)
                1 -> getString(R.string.class_low_vigilant)
                else -> getString(R.string.class_drowsy)
            }
            getString(R.string.hud_class, name, p[0], p[1], p[2])
        } else getString(R.string.hud_class_na)

        val alertText = when {
            AlertType.EYES_CLOSED in out.alerts -> getString(R.string.alert_eyes_closed)
            AlertType.DROWSY in out.alerts -> getString(R.string.alert_drowsy)
            AlertType.PERCLOS in out.alerts -> getString(R.string.alert_perclos)
            else -> null
        }
        binding.txtAlert.visibility = if (alertText != null) View.VISIBLE else View.GONE
        if (alertText != null) binding.txtAlert.text = alertText
        binding.alertOverlay.active = alertText != null

        benchmark?.let { b ->
            if (!b.finished) binding.btnBenchmark.text = getString(R.string.btn_benchmark_running, (b.elapsedMs / 1000).toInt())
        }
    }

    // ------------------------------------------------------------------ controle

    private fun startEngine() {
        if (landmarker == null) {
            Toast.makeText(this, R.string.landmarker_missing, Toast.LENGTH_LONG).show()
            return
        }
        synchronized(engineLock) { engine.start() }
        latLandmarker.reset(); latCnn.reset(); latTotal.reset()
        running = true
        binding.btnStart.setText(R.string.btn_stop)
    }

    private fun stopEngine() {
        running = false
        synchronized(engineLock) { engine.stop() }
        benchmark?.let { finishBenchmark(it) }
        binding.btnStart.setText(R.string.btn_start)
        binding.txtStatus.setText(R.string.status_idle)
        binding.txtAlert.visibility = View.GONE
        binding.alertOverlay.active = false
    }

    private fun startBenchmark() {
        if (!running) {
            Toast.makeText(this, R.string.benchmark_need_running, Toast.LENGTH_SHORT).show()
            return
        }
        if (benchmark?.finished == false) return
        benchmark = BenchmarkRecorder(this, 60_000L)
        binding.btnBenchmark.text = getString(R.string.btn_benchmark_running, 0)
    }

    private fun finishBenchmark(b: BenchmarkRecorder) {
        if (b.finished) return
        val json = b.finish(landmarker?.delegateName, modelSizes(), eyeModelAsset,
            if (temporalModel != null) BlinkTemporalModel.ASSET else null)
        lastBenchmarkFiles = listOf(b.csvFile, json)
        benchmark = null
        mainHandler.post {
            binding.btnBenchmark.setText(R.string.btn_benchmark)
            Toast.makeText(this, getString(R.string.benchmark_saved, b.dir.absolutePath), Toast.LENGTH_LONG).show()
        }
    }

    private fun shareBenchmark() {
        val files = lastBenchmarkFiles.ifEmpty {
            BenchmarkRecorder.dirFor(this).listFiles()?.sortedByDescending { it.lastModified() }?.take(2) ?: emptyList()
        }.filter { it.exists() && it.length() > 0 }
        if (files.isEmpty()) {
            Toast.makeText(this, R.string.share_nothing, Toast.LENGTH_SHORT).show()
            return
        }
        val uris = ArrayList<Uri>(files.map { FileProvider.getUriForFile(this, "$packageName.fileprovider", it) })
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "*/*"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, getString(R.string.share_title)))
    }

    override fun onDestroy() {
        super.onDestroy()
        running = false
        cameraExecutor.shutdown()
        landmarker?.close()
        temporalModel?.close()
        eyeModel?.close()
        alerter.release()
    }

    companion object {
        const val TAG = "MainActivity"
    }
}
