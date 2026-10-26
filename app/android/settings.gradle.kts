// Projeto Android do TCC "Detecção de Sonolência em Dispositivos Móveis" (PUCPR, Equipe 14).
// Abra ESTA pasta (app/android) no Android Studio.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()        // AndroidX, CameraX, MediaPipe Tasks, LiteRT
        mavenCentral()  // TensorFlow Lite (shim), kotlinx-coroutines, JUnit
    }
}

rootProject.name = "SonolenciaTCC"
include(":app")
