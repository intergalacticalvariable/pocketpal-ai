import type {TurboModule} from 'react-native';
import {Platform, TurboModuleRegistry} from 'react-native';

export interface LiteRTLoadOptions {
  /** 'cpu' or 'gpu'. */
  backend: string;
  /** KV-cache size: the sum of input and output tokens. */
  maxNumTokens?: number;
  threadCount?: number;
}

export interface LiteRTLoadResult {
  contextId: number;
  backend: string;
}

export interface LiteRTChatMessage {
  /** 'user' or 'model'. */
  role: string;
  text: string;
}

export interface LiteRTGenerateParams {
  systemInstruction?: string;
  /** Prior turns; the last user message goes in `prompt`. */
  history: LiteRTChatMessage[];
  prompt: string;
  topK: number;
  topP: number;
  temperature: number;
  seed: number;
}

export interface LiteRTGenerateResult {
  text: string;
  interrupted: boolean;
}

export interface Spec extends TurboModule {
  addListener(eventName: string): void;
  removeListeners(count: number): void;

  loadModel(
    path: string,
    options: LiteRTLoadOptions,
  ): Promise<LiteRTLoadResult>;
  /**
   * Runs one turn on a fresh conversation seeded with the history, streaming
   * chunks as `onLiteRTToken` events: {contextId, requestId, delta}.
   */
  generate(
    contextId: number,
    requestId: string,
    params: LiteRTGenerateParams,
  ): Promise<LiteRTGenerateResult>;
  stopGeneration(contextId: number): Promise<void>;
  release(contextId: number): Promise<void>;
}

// LiteRT-LM is wired up on Android only.
export default Platform.OS === 'android'
  ? TurboModuleRegistry.get<Spec>('LiteRTLMModule')
  : null;
