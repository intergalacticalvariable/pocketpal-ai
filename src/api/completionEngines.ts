import {NativeEventEmitter} from 'react-native';
import {LlamaContext} from 'llama.rn';

import {streamChatCompletion} from './openai';
import type {RemoteEndpoint} from './servers';
import {pickSamplers} from '../utils/samplerParams';
import {
  ApiCompletionParams,
  CompletionEngine,
  CompletionResult,
  CompletionStreamData,
  normaliseTimings,
} from '../utils/completionTypes';
import {RemoteModelRequestWithdrawnError} from '../utils/errors';
import {isModelNotLoaded} from './llamaServer/routerWire';
import NativeLiteRTLM from '../specs/NativeLiteRTLM';
import type {
  LiteRTChatMessage,
  LiteRTGenerateParams,
  Spec as LiteRTSpec,
} from '../specs/NativeLiteRTLM';

export class LocalCompletionEngine implements CompletionEngine {
  constructor(private context: LlamaContext) {}

  async completion(
    params: ApiCompletionParams,
    callback?: (data: CompletionStreamData) => void,
  ): Promise<CompletionResult> {
    const result = await this.context.completion(
      params,
      callback
        ? data => {
            callback({
              token: data.token,
              content: data.content,
              reasoning_content: data.reasoning_content,
              tool_calls: data.tool_calls,
              accumulated_text: data.accumulated_text,
            });
          }
        : undefined,
    );
    return {
      text: result.text,
      content: result.content,
      reasoning_content: result.reasoning_content,
      tool_calls: result.tool_calls,
      timings: normaliseTimings(result.timings),
      tokens_predicted: result.tokens_predicted,
      tokens_evaluated: result.tokens_evaluated,
      draft_tokens: result.draft_tokens,
      draft_tokens_accepted: result.draft_tokens_accepted,
      truncated: result.truncated,
      stopped_eos: result.stopped_eos,
      stopped_limit: result.stopped_limit,
      stopped_word: result.stopped_word,
      stopping_word: result.stopping_word,
      context_full: result.context_full,
      interrupted: result.interrupted,
    };
  }

  async stopCompletion(): Promise<void> {
    await this.context.stopCompletion();
  }
}

export interface RemoteEngineOptions {
  /**
   * Awaited before the request, with the signal Stop aborts. The engine
   * applies no timeout of its own: whoever supplies this bounds it. With
   * `reread`, the cached list is not trusted; resolves true only when a
   * router made the model ready.
   */
  ensureReady?: (
    signal: AbortSignal,
    options?: {reread: boolean},
  ) => Promise<boolean | void>;
}

export class OpenAICompletionEngine implements CompletionEngine {
  private abortController: AbortController | null = null;

  constructor(
    private endpoint: RemoteEndpoint,
    private options: RemoteEngineOptions = {},
  ) {}

  async completion(
    params: ApiCompletionParams,
    callback?: (data: CompletionStreamData) => void,
  ): Promise<CompletionResult> {
    const controller = new AbortController();
    this.abortController = controller;

    const {ensureReady} = this.options;
    if (!ensureReady) {
      return this.send(params, controller.signal, callback);
    }
    await ensureReady(controller.signal);
    if (controller.signal.aborted) {
      throw new RemoteModelRequestWithdrawnError();
    }

    let streamed = false;
    try {
      return await this.send(params, controller.signal, data => {
        streamed = true;
        callback?.(data);
      });
    } catch (error) {
      if (streamed || !isModelNotLoaded(error)) {
        throw error;
      }
      const ready = await ensureReady(controller.signal, {reread: true});
      if (controller.signal.aborted) {
        throw new RemoteModelRequestWithdrawnError();
      }
      if (ready !== true) {
        throw error;
      }
      return this.send(params, controller.signal, callback);
    }
  }

  private send(
    params: ApiCompletionParams,
    signal: AbortSignal,
    callback?: (data: CompletionStreamData) => void,
  ): Promise<CompletionResult> {
    return streamChatCompletion(
      {
        messages: params.messages || [],
        model: this.endpoint.remoteModelId,
        samplers: pickSamplers(params),
        stop: params.stop,
        stream: true,
        // llama.rn's `tools` typedef is structurally compatible with OpenAI's
        // function-tool shape but lives under a different name.
        tools: (params as any).tools,
        tool_choice: (params as any).tool_choice,
        response_format: (params as any).response_format,
        // Reasoning intent carried on the params; the server profile owns the wire shape.
        reasoning: params.reasoning,
      },
      this.endpoint,
      signal,
      callback,
    );
  }

  async stopCompletion(): Promise<void> {
    this.abortController?.abort();
    this.abortController = null;
  }
}

/** The file extension LiteRT-LM model bundles carry. */
export const LITERT_MODEL_EXTENSION = '.litertlm';

export const isLiteRTModelFile = (filename: string | undefined): boolean =>
  !!filename && filename.toLowerCase().endsWith(LITERT_MODEL_EXTENSION);

const LITERT_TOKEN_EVENT = 'onLiteRTToken';

type ApiMessage = NonNullable<ApiCompletionParams['messages']>[number];

const messageText = (content: ApiMessage['content']): string => {
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return '';
  }
  return content
    .map(part =>
      part && part.type === 'text' && typeof part.text === 'string'
        ? part.text
        : '',
    )
    .join('');
};

/**
 * Maps the OpenAI-shaped messages onto a LiteRT-LM conversation: system
 * messages join into the system instruction, the last user message becomes the
 * prompt and everything before it the seeded history. LiteRT-LM has only user
 * and model turns here, so a tool result is replayed as user text.
 */
export function toLiteRTRequest(
  messages: ApiMessage[],
): Pick<LiteRTGenerateParams, 'systemInstruction' | 'history' | 'prompt'> {
  const system: string[] = [];
  const turns: LiteRTChatMessage[] = [];
  for (const message of messages) {
    const text = messageText(message.content);
    switch (message.role) {
      case 'system':
        if (text) {
          system.push(text);
        }
        break;
      case 'assistant':
        if (text) {
          turns.push({role: 'model', text});
        }
        break;
      case 'tool':
        turns.push({role: 'user', text: `Tool result:\n${text}`});
        break;
      default:
        turns.push({role: 'user', text});
    }
  }
  const last = turns[turns.length - 1];
  const prompt =
    last?.role === 'user' ? (turns.pop() as LiteRTChatMessage) : undefined;
  return {
    systemInstruction: system.length > 0 ? system.join('\n\n') : undefined,
    history: turns,
    prompt: prompt?.text ?? '',
  };
}

const positiveOr = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0
    ? value
    : fallback;

/**
 * Runs completions on a model loaded by the LiteRT-LM native module (Android).
 * Each turn replays the chat on a fresh native conversation, so the engine
 * holds no state between turns beyond the loaded model.
 */
export class LiteRTCompletionEngine implements CompletionEngine {
  private static nextRequest = 0;
  private emitter: NativeEventEmitter;

  constructor(
    readonly contextId: number,
    private native: LiteRTSpec = NativeLiteRTLM as LiteRTSpec,
  ) {
    if (!native) {
      throw new Error('LiteRT-LM is not available on this platform');
    }
    this.emitter = new NativeEventEmitter(native as any);
  }

  async completion(
    params: ApiCompletionParams,
    callback?: (data: CompletionStreamData) => void,
  ): Promise<CompletionResult> {
    const requestId = `${this.contextId}-${++LiteRTCompletionEngine.nextRequest}`;
    const stops = (params.stop ?? []).filter(s => !!s);
    let accumulated = '';
    let visible = '';
    let stoppingWord: string | undefined;
    const startedAt = Date.now();

    const subscription = this.emitter.addListener(
      LITERT_TOKEN_EVENT,
      (event: {requestId?: string; delta?: string}) => {
        if (event.requestId !== requestId || stoppingWord !== undefined) {
          return;
        }
        const delta = event.delta ?? '';
        accumulated += delta;
        for (const stop of stops) {
          const at = accumulated.indexOf(stop);
          if (at !== -1) {
            stoppingWord = stop;
            accumulated = accumulated.slice(0, at);
          }
        }
        const token = accumulated.slice(visible.length);
        visible = accumulated;
        if (stoppingWord !== undefined) {
          this.native.stopGeneration(this.contextId).catch(() => {});
        }
        if (token && callback) {
          callback({token, content: visible, accumulated_text: visible});
        }
      },
    );

    let result;
    try {
      result = await this.native.generate(this.contextId, requestId, {
        ...toLiteRTRequest(params.messages ?? []),
        topK: Math.round(positiveOr(params.top_k, 40)),
        topP: Math.min(positiveOr(params.top_p, 0.95), 1),
        temperature:
          typeof params.temperature === 'number' && params.temperature >= 0
            ? params.temperature
            : 0.7,
        seed:
          typeof params.seed === 'number' && params.seed >= 0
            ? Math.floor(params.seed)
            : 0,
      });
    } finally {
      subscription.remove();
    }

    const text =
      stoppingWord !== undefined ? accumulated : accumulated || result.text;
    const interrupted = result.interrupted && stoppingWord === undefined;
    const elapsedMs = Date.now() - startedAt;
    return {
      text,
      content: text,
      timings: elapsedMs > 0 ? {predicted_ms: elapsedMs} : undefined,
      stopped_eos: !interrupted && stoppingWord === undefined,
      stopped_word: stoppingWord,
      stopping_word: stoppingWord,
      interrupted,
    };
  }

  async stopCompletion(): Promise<void> {
    await this.native.stopGeneration(this.contextId);
  }

  async release(): Promise<void> {
    await this.native.release(this.contextId);
  }
}
