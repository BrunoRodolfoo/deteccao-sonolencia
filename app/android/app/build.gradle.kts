import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "br.pucpr.sonolencia"
    // CameraX 1.6.2, core-ktx 1.17.0 e activity 1.12.4 exigem compileSdk >= 36 (aar-metadata.properties).
    compileSdk = 36

    defaultConfig {
        applicationId = "br.pucpr.sonolencia"
        minSdk = 26          // Android 8.0 (necessário para VibrationEffect e adaptive icons)
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        // O Galaxy S23 Ultra (Snapdragon 8 Gen 2) é arm64; manter só este ABI reduz o APK
        // (as bibliotecas nativas do MediaPipe e do LiteRT somam ~15 MB por ABI).
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        viewBinding = true
    }

    // Modelos devem ficar sem compressão no APK para que o LiteRT/MediaPipe possam mapeá-los
    // em memória (mmap) diretamente do AssetFileDescriptor.
    androidResources {
        noCompress += listOf("tflite", "task")
    }

    packaging {
        resources {
            excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1", "META-INF/DEPENDENCIES")
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // AndroidX básico (versões estáveis verificadas em dl.google.com/dl/android/maven2, 20/09/2026)
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.activity:activity-ktx:1.12.4")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")

    // CameraX 1.6.2 (estável, 26/08/2026)
    val cameraxVersion = "1.6.2"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // MediaPipe Tasks Vision (FaceLandmarker). 0.10.35 é a última da série 0.10.x, usada pelos
    // exemplos oficiais; 1.0.0 também existe no Maven do Google e expõe a mesma API.
    implementation("com.google.mediapipe:tasks-vision:0.10.35")

    // TensorFlow Lite. 2.17.0 é a última versão publicada em org.tensorflow no Maven Central;
    // ela delega para com.google.ai.edge.litert:litert:1.0.1 (Maven do Google), que ainda
    // expõe o pacote org.tensorflow.lite (Interpreter, Tensor, DataType).
    implementation("org.tensorflow:tensorflow-lite:2.17.0")

    // Coroutines (1.10.2 é estável e compatível com Kotlin 2.2.x)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    // Testes unitários JVM puros (sem emulador) para DrowsinessEngine e utilitários.
    testImplementation("junit:junit:4.13.2")
}
