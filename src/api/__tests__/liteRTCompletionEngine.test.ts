import {NativeEventEmitter} from 'react-native';

import {
  LiteRTCompletionEngine,
  isLiteRTModelFile,
  toLiteRTRequest,
} from '../completionEngines';
import type {Spec} from '../../specs/NativeLiteRTLM';

type TokenListener = (event: {requestId?: string; delta?: string}) => void;

describe('isLiteRTModelFile', () => {
  it('matches .litertlm bundles only', () => {
    expect(isLiteRTModelFile('gemma-4-E2B-it.litertlm')).toBe(true);
    expect(isLiteRTModelFile('GEMMA.LITERTLM')).toBe(true);
    expect(isLiteRTModelFile('gemma-4-E2B-it-Q4_0.gguf')).toBe(false);
    expect(isLiteRTModelFile(undefined)).toBe(false);
  });
});

describe('toLiteRTRequest', () => {
  it('splits system, history and the last user prompt', () => {
    expect(
      toLiteRTRequest([
        {role: 'system', content: 'Be brief.'},
        {role: 'system', content: 'Answer in English.'},
        {role: 'user', content: 'Hi'},
        {role: 'assistant', content: 'Hello!'},
        {
          role: 'user',
          content: [
            {type: 'text', text: 'What is '},
            {type: 'image_url', image_url: {url: 'file://x.png'}},
            {type: 'text', text: '2+2?'},
          ],
        },
      ] as any),
    ).toEqual({
      systemInstruction: 'Be brief.\n\nAnswer in English.',
      history: [
        {role: 'user', text: 'Hi'},
        {role: 'model', text: 'Hello!'},
      ],
      prompt: 'What is 2+2?',
    });
  });

  it('replays tool results as user text and skips empty model turns', () => {
    expect(
      toLiteRTRequest([
        {role: 'user', content: 'Search'},
        {role: 'assistant', content: ''},
        {role: 'tool', content: '{"ok":true}'},
        {role: 'user', content: 'Thanks'},
      ] as any),
    ).toEqual({
      systemInstruction: undefined,
      history: [
        {role: 'user', text: 'Search'},
        {role: 'user', text: 'Tool result:\n{"ok":true}'},
      ],
      prompt: 'Thanks',
    });
  });
});

describe('LiteRTCompletionEngine', () => {
  let listener: TokenListener | undefined;
  let remove: jest.Mock;
  let native: jest.Mocked<Spec>;

  const emit = (requestId: string, delta: string) =>
    listener?.({requestId, delta});

  beforeEach(() => {
    listener = undefined;
    remove = jest.fn();
    (NativeEventEmitter.prototype.addListener as jest.Mock).mockImplementation(
      (_event: string, cb: TokenListener) => {
        listener = cb;
        return {remove};
      },
    );
    native = {
      addListener: jest.fn(),
      removeListeners: jest.fn(),
      loadModel: jest.fn(),
      generate: jest.fn(),
      stopGeneration: jest.fn().mockResolvedValue(undefined),
      release: jest.fn().mockResolvedValue(undefined),
    } as unknown as jest.Mocked<Spec>;
  });

  it('streams chunks for its own request and maps samplers', async () => {
    native.generate.mockImplementation(async (_id, requestId) => {
      emit('other-request', 'ignored');
      emit(requestId, 'Hel');
      emit(requestId, 'lo');
      return {text: 'Hello', interrupted: false};
    });
    const engine = new LiteRTCompletionEngine(7, native);
    const chunks: string[] = [];

    const result = await engine.completion(
      {
        messages: [{role: 'user', content: 'Hi'}],
        temperature: 0.2,
        top_k: 20,
        top_p: 0.9,
        seed: -1,
      } as any,
      data => chunks.push(data.token ?? ''),
    );

    expect(native.generate).toHaveBeenCalledWith(7, expect.any(String), {
      systemInstruction: undefined,
      history: [],
      prompt: 'Hi',
      topK: 20,
      topP: 0.9,
      temperature: 0.2,
      seed: 0,
    });
    expect(chunks).toEqual(['Hel', 'lo']);
    expect(result).toMatchObject({
      text: 'Hello',
      content: 'Hello',
      stopped_eos: true,
      interrupted: false,
    });
    expect(remove).toHaveBeenCalled();
  });

  it('cuts at a stop word and stops the native turn', async () => {
    native.generate.mockImplementation(async (_id, requestId) => {
      emit(requestId, 'Answer.<end');
      emit(requestId, '_of_turn> more');
      return {text: 'Answer.<end_of_turn> more', interrupted: true};
    });
    const engine = new LiteRTCompletionEngine(1, native);

    const result = await engine.completion({
      messages: [{role: 'user', content: 'Q'}],
      stop: ['<end_of_turn>'],
    } as any);

    expect(native.stopGeneration).toHaveBeenCalledWith(1);
    expect(result).toMatchObject({
      text: 'Answer.',
      stopped_word: '<end_of_turn>',
      stopped_eos: false,
      interrupted: false,
    });
  });

  it('reports a user stop as interrupted', async () => {
    native.generate.mockResolvedValue({text: 'Part', interrupted: true});
    const engine = new LiteRTCompletionEngine(1, native);

    const result = await engine.completion({
      messages: [{role: 'user', content: 'Q'}],
    } as any);

    expect(result).toMatchObject({
      text: 'Part',
      interrupted: true,
      stopped_eos: false,
    });
  });

  it('removes its listener when generation fails', async () => {
    native.generate.mockRejectedValue(new Error('boom'));
    const engine = new LiteRTCompletionEngine(1, native);

    await expect(
      engine.completion({messages: [{role: 'user', content: 'Q'}]} as any),
    ).rejects.toThrow('boom');
    expect(remove).toHaveBeenCalled();
  });

  it('forwards stop and release to the native context', async () => {
    const engine = new LiteRTCompletionEngine(3, native);
    await engine.stopCompletion();
    await engine.release();
    expect(native.stopGeneration).toHaveBeenCalledWith(3);
    expect(native.release).toHaveBeenCalledWith(3);
  });
});
