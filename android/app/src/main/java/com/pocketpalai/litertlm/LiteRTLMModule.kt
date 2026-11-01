package com.pocketpal.litertlm

import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.module.annotations.ReactModule
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.pocketpal.specs.NativeLiteRTLMSpec
import java.util.concurrent.CancellationException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Runs `.litertlm` model bundles (e.g. Gemma 4 E2B/E4B) on Google's LiteRT-LM.
 *
 * Each `generate` call replays the chat on a fresh Conversation and closes it
 * when the turn ends, so the only state kept between turns is the loaded
 * Engine. Chunks stream to JS as `onLiteRTToken` events tagged with the
 * caller's requestId.
 */
@ReactModule(name = NativeLiteRTLMSpec.NAME)
class LiteRTLMModule(reactContext: ReactApplicationContext) :
    NativeLiteRTLMSpec(reactContext) {

  private class ActiveTurn(val conversation: Conversation) {
    val cancelled = AtomicBoolean(false)
  }

  // Engine init, conversation setup and teardown are blocking native calls;
  // one worker keeps them off the JS thread and ordered.
  private val worker = Executors.newSingleThreadExecutor()
  private val engines = ConcurrentHashMap<Int, Engine>()
  private val turns = ConcurrentHashMap<Int, ActiveTurn>()
  private val nextContextId = AtomicInteger(1)

  override fun addListener(eventName: String) {}

  override fun removeListeners(count: Double) {}

  override fun loadModel(path: String, options: ReadableMap, promise: Promise) {
    worker.execute {
      try {
        val wantsGpu = options.getStringOrNull("backend") == "gpu"
        val maxNumTokens = options.getIntOrNull("maxNumTokens")
        val threadCount = options.getIntOrNull("threadCount")?.takeIf { it > 0 }

        var backendName = if (wantsGpu) "gpu" else "cpu"
        val engine =
            try {
              initEngine(path, if (wantsGpu) Backend.GPU() else Backend.CPU(threadCount, null), maxNumTokens)
            } catch (e: Exception) {
              if (!wantsGpu) throw e
              Log.w(TAG, "GPU init failed, falling back to CPU", e)
              backendName = "cpu"
              initEngine(path, Backend.CPU(threadCount, null), maxNumTokens)
            }

        val contextId = nextContextId.getAndIncrement()
        engines[contextId] = engine
        promise.resolve(
            Arguments.createMap().apply {
              putInt("contextId", contextId)
              putString("backend", backendName)
            })
      } catch (e: Throwable) {
        Log.e(TAG, "Failed to load $path", e)
        promise.reject("LITERT_LOAD_FAILED", e.message ?: "Failed to load model", e)
      }
    }
  }

  private fun initEngine(path: String, backend: Backend, maxNumTokens: Int?): Engine {
    val config =
        EngineConfig(
            path,
            backend,
            null, // visionBackend: text only for now
            null, // audioBackend
            maxNumTokens,
            null, // maxNumImages
            reactApplicationContext.cacheDir.absolutePath,
        )
    val engine = Engine(config)
    engine.initialize()
    return engine
  }

  override fun generate(
      contextId: Double,
      requestId: String,
      params: ReadableMap,
      promise: Promise,
  ) {
    val id = contextId.toInt()
    worker.execute {
      val engine = engines[id]
      if (engine == null) {
        promise.reject("LITERT_NO_CONTEXT", "No LiteRT-LM model loaded for context $id")
        return@execute
      }
      try {
        // A turn still running on this engine is superseded.
        turns.remove(id)?.let { cancelAndClose(it) }

        val conversation = engine.createConversation(conversationConfig(params))
        val turn = ActiveTurn(conversation)
        turns[id] = turn

        val text = StringBuilder()
        val settled = AtomicBoolean(false)
        val finish = { interrupted: Boolean, error: Throwable? ->
          if (settled.compareAndSet(false, true)) {
            turns.remove(id, turn)
            worker.execute { closeQuietly(conversation) }
            if (error != null) {
              promise.reject("LITERT_GENERATE_FAILED", error.message ?: "Generation failed", error)
            } else {
              promise.resolve(
                  Arguments.createMap().apply {
                    putString("text", text.toString())
                    putBoolean("interrupted", interrupted)
                  })
            }
          }
        }

        conversation.sendMessageAsync(
            params.getString("prompt") ?: "",
            object : MessageCallback {
              override fun onMessage(message: Message) {
                val delta = textOf(message)
                if (delta.isEmpty()) return
                text.append(delta)
                emitToken(id, requestId, delta)
              }

              override fun onDone() {
                finish(turn.cancelled.get(), null)
              }

              override fun onError(throwable: Throwable) {
                if (turn.cancelled.get() || throwable is CancellationException) {
                  finish(true, null)
                } else {
                  finish(false, throwable)
                }
              }
            })
      } catch (e: Throwable) {
        turns.remove(id)?.let { closeQuietly(it.conversation) }
        Log.e(TAG, "Generation failed to start", e)
        promise.reject("LITERT_GENERATE_FAILED", e.message ?: "Generation failed", e)
      }
    }
  }

  private fun conversationConfig(params: ReadableMap): ConversationConfig {
    val system = params.getStringOrNull("systemInstruction")?.takeIf { it.isNotEmpty() }
    val history = mutableListOf<Message>()
    params.getArray("history")?.let { array ->
      for (i in 0 until array.size()) {
        val entry = array.getMap(i) ?: continue
        val text = entry.getString("text") ?: ""
        history.add(if (entry.getString("role") == "model") Message.model(text) else Message.user(text))
      }
    }
    val sampler =
        SamplerConfig(
            params.getIntOrNull("topK")?.coerceAtLeast(1) ?: 40,
            params.getDoubleOrNull("topP")?.coerceIn(0.0, 1.0) ?: 0.95,
            params.getDoubleOrNull("temperature")?.coerceAtLeast(0.0) ?: 0.7,
            params.getIntOrNull("seed") ?: 0,
        )
    return ConversationConfig(
        system?.let { Contents.of(it) },
        history,
        emptyList(),
        sampler,
    )
  }

  override fun stopGeneration(contextId: Double, promise: Promise) {
    turns[contextId.toInt()]?.let { turn ->
      turn.cancelled.set(true)
      try {
        turn.conversation.cancelProcess()
      } catch (e: Exception) {
        Log.w(TAG, "cancelProcess failed", e)
      }
    }
    promise.resolve(null)
  }

  override fun release(contextId: Double, promise: Promise) {
    val id = contextId.toInt()
    worker.execute {
      try {
        turns.remove(id)?.let { cancelAndClose(it) }
        engines.remove(id)?.close()
        promise.resolve(null)
      } catch (e: Throwable) {
        promise.reject("LITERT_RELEASE_FAILED", e.message ?: "Release failed", e)
      }
    }
  }

  override fun invalidate() {
    worker.execute {
      turns.values.forEach { cancelAndClose(it) }
      turns.clear()
      engines.values.forEach { engine ->
        try {
          engine.close()
        } catch (e: Exception) {
          Log.w(TAG, "Engine close failed", e)
        }
      }
      engines.clear()
    }
    worker.shutdown()
    super.invalidate()
  }

  private fun cancelAndClose(turn: ActiveTurn) {
    turn.cancelled.set(true)
    try {
      turn.conversation.cancelProcess()
    } catch (e: Exception) {
      Log.w(TAG, "cancelProcess failed", e)
    }
    closeQuietly(turn.conversation)
  }

  private fun closeQuietly(conversation: Conversation) {
    try {
      conversation.close()
    } catch (e: Exception) {
      // Already closed.
    }
  }

  private fun emitToken(contextId: Int, requestId: String, delta: String) {
    val event =
        Arguments.createMap().apply {
          putInt("contextId", contextId)
          putString("requestId", requestId)
          putString("delta", delta)
        }
    reactApplicationContext
        .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
        .emit(TOKEN_EVENT, event)
  }

  // Only the visible text: reasoning arrives on the message's channels.
  private fun textOf(message: Message): String =
      message.contents.contents.joinToString("") { if (it is Content.Text) it.text else "" }

  private fun ReadableMap.getStringOrNull(key: String): String? =
      if (hasKey(key) && !isNull(key)) getString(key) else null

  private fun ReadableMap.getIntOrNull(key: String): Int? =
      if (hasKey(key) && !isNull(key)) getDouble(key).toInt() else null

  private fun ReadableMap.getDoubleOrNull(key: String): Double? =
      if (hasKey(key) && !isNull(key)) getDouble(key) else null

  companion object {
    private const val TAG = "LiteRTLMModule"
    private const val TOKEN_EVENT = "onLiteRTToken"
  }
}
