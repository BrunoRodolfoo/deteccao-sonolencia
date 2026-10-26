# Regras ProGuard/R8 (minify está desativado por padrão; regras mantidas por segurança).
# MediaPipe Tasks e LiteRT usam JNI: preservar classes públicas.
-keep class com.google.mediapipe.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-dontwarn org.tensorflow.**
-dontwarn com.google.mediapipe.**
