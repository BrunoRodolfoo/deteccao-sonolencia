package br.pucpr.sonolencia

import kotlin.math.sqrt

/**
 * Geometria facial pura (sem dependência de Android/MediaPipe), idêntica ao protótipo
 * Python `src/realtime/metrics.py`.
 *
 * Índices dos landmarks do MediaPipe Face Mesh / Face Landmarker (478 pontos; os 468
 * primeiros coincidem com o Face Mesh). Ordem p1..p6 da fórmula do EAR: canto externo,
 * dois pontos da pálpebra superior, canto interno, dois pontos da pálpebra inferior.
 */
object FaceMetrics {
    val LEFT_EYE = intArrayOf(362, 385, 387, 263, 373, 380)
    val RIGHT_EYE = intArrayOf(33, 160, 158, 133, 153, 144)

    /** Boca: cantos (61, 291) e par vertical interno (13, 14). */
    val MOUTH_CORNERS = intArrayOf(61, 291)
    val MOUTH_VERTICAL = intArrayOf(13, 14)

    const val NUM_LANDMARKS = 478

    data class Point(val x: Float, val y: Float)

    /** Caixa quadrada em torno de um olho, em pixels (para o recorte da CNN). */
    data class EyeBox(val cx: Float, val cy: Float, val side: Float)

    fun dist(a: Point, b: Point): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)  (Soukupová & Čech, 2016).
     * Mesma forma usada por Ghoddoosian et al. (blink_video.py) e pelo protótipo.
     * Olho aberto ~0.25-0.35; fechado tende a ~0.
     */
    fun eyeAspectRatio(p: List<Point>): Float {
        require(p.size == 6) { "EAR requer 6 pontos" }
        val horiz = dist(p[0], p[3])
        if (horiz < 1e-6f) return 0f
        return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2f * horiz)
    }

    /** MAR: abertura vertical interna da boca normalizada pela largura entre os cantos. */
    fun mouthAspectRatio(corner1: Point, corner2: Point, top: Point, bottom: Point): Float {
        val horiz = dist(corner1, corner2)
        if (horiz < 1e-6f) return 0f
        return dist(top, bottom) / horiz
    }

    /**
     * Landmarks em pixels como vetor plano [x0, y0, x1, y1, ...]. Recebe as coordenadas
     * normalizadas do MediaPipe (0..1) e as dimensões da imagem analisada.
     */
    class Landmarks(val xy: FloatArray) {
        init {
            require(xy.size >= NUM_LANDMARKS * 2) { "Esperados >= ${NUM_LANDMARKS * 2} valores" }
        }

        fun point(i: Int) = Point(xy[2 * i], xy[2 * i + 1])
        fun points(idx: IntArray): List<Point> = idx.map { point(it) }

        fun leftEar() = eyeAspectRatio(points(LEFT_EYE))
        fun rightEar() = eyeAspectRatio(points(RIGHT_EYE))
        fun mar() = mouthAspectRatio(
            point(MOUTH_CORNERS[0]), point(MOUTH_CORNERS[1]),
            point(MOUTH_VERTICAL[0]), point(MOUTH_VERTICAL[1])
        )

        /**
         * Caixa quadrada centrada no centro do olho (média dos 6 pontos), lado =
         * [scale] x distância entre p1 e p4 (cantos). Padrão 1.6: aproxima a caixa de
         * 0,35 x largura da face usada nos recortes "crop" do CEW no treino
         * (src/pipeline_b/datasets.py, CEW_CROP_FRAC); ver [EyeCropper].
         */
        fun eyeBox(idx: IntArray, scale: Float = 1.6f): EyeBox {
            val pts = points(idx)
            val cx = pts.map { it.x }.average().toFloat()
            val cy = pts.map { it.y }.average().toFloat()
            val side = scale * dist(pts[0], pts[3])
            return EyeBox(cx, cy, side)
        }
    }
}
