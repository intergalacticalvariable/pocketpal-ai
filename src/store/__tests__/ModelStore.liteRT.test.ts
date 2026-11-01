jest.unmock('../../store');
import {runInAction} from 'mobx';
import {initLlama} from 'llama.rn';
import {Platform} from 'react-native';

import {Model, ModelOrigin} from '../../utils/types';
import {createModel} from '../../../jest/fixtures/models';
import {LiteRTCompletionEngine} from '../../api/completionEngines';
import type {Tier} from '../../services/deviceRules/types';
import {liteRTPresetCandidates} from '../liteRTPresets';

import {modelStore} from '..';

jest.mock('../../specs/NativeLiteRTLM', () => ({
  __esModule: true,
  default: {
    addListener: jest.fn(),
    removeListeners: jest.fn(),
    loadModel: jest.fn(),
    generate: jest.fn(),
    stopGeneration: jest.fn(),
    release: jest.fn(),
  },
}));

const mockNative = jest.requireMock('../../specs/NativeLiteRTLM')
  .default as Record<string, jest.Mock>;

jest.mock('../../hooks/useMemoryCheck', () => ({
  hasEnoughMemory: jest.fn().mockResolvedValue(true),
}));

jest.mock('../../services/deviceRules/rules', () => ({
  fetchRules: jest.fn().mockResolvedValue(null),
}));
jest.mock('../../services/deviceRules/signals', () => ({
  readDeviceSignals: jest
    .fn()
    .mockResolvedValue({ramBytes: 8 * 1e9, socModel: 'SM8850'}),
}));

const liteRTModel = createModel({
  id: 'litert-1',
  name: 'gemma-4-E2B-it.litertlm',
  filename: 'gemma-4-E2B-it.litertlm',
  origin: ModelOrigin.LOCAL,
  isLocal: true,
  isDownloaded: true,
  downloadUrl: '',
  fullPath: '/models/gemma-4-E2B-it.litertlm',
  size: 2_580_000_000,
}) as Model;

describe('ModelStore with a LiteRT-LM model', () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    mockNative.loadModel.mockResolvedValue({contextId: 5, backend: 'gpu'});
    mockNative.stopGeneration.mockResolvedValue(undefined);
    mockNative.release.mockResolvedValue(undefined);
    await modelStore.releaseContext(true);
    runInAction(() => {
      modelStore.models = [liteRTModel];
    });
  });

  it('loads on the native module instead of llama.rn', async () => {
    const result = await modelStore.initContext(liteRTModel);

    expect(result).toBeNull();
    expect(initLlama).not.toHaveBeenCalled();
    expect(mockNative.loadModel).toHaveBeenCalledWith(
      '/models/gemma-4-E2B-it.litertlm',
      expect.objectContaining({
        backend: expect.stringMatching(/^(cpu|gpu)$/),
        maxNumTokens: modelStore.contextInitParams.n_ctx,
      }),
    );
    expect(modelStore.engine).toBeInstanceOf(LiteRTCompletionEngine);
    expect(modelStore.context).toBeUndefined();
    expect(modelStore.activeModelId).toBe('litert-1');
    expect(modelStore.contextId).toBe('litert-5');
  });

  it('does not reload a model that is already loaded', async () => {
    await modelStore.initContext(liteRTModel);
    await modelStore.initContext(liteRTModel);
    expect(mockNative.loadModel).toHaveBeenCalledTimes(1);
  });

  it('frees the native model on release', async () => {
    await modelStore.initContext(liteRTModel);
    await modelStore.releaseContext(true);

    expect(mockNative.release).toHaveBeenCalledWith(5);
    expect(modelStore.engine).toBeUndefined();
    expect(modelStore.activeModelId).toBeUndefined();
  });

  it('records a load error and rethrows it', async () => {
    mockNative.loadModel.mockRejectedValue(new Error('bad bundle'));

    await expect(modelStore.initContext(liteRTModel)).rejects.toThrow(
      'bad bundle',
    );
    expect(modelStore.modelLoadError).toBeDefined();
    expect(modelStore.engine).toBeUndefined();
  });
});

describe('LiteRT-LM presets', () => {
  it('offers Gemma 4 E2B from the mid tier and E4B from the high tier, Android only', () => {
    const names = (tier: Tier, os: string) =>
      liteRTPresetCandidates(tier, os).map(c => c.hfFilename);

    expect(names('low', 'android')).toEqual([]);
    expect(names('mid', 'android')).toEqual(['gemma-4-E2B-it.litertlm']);
    expect(names('high', 'android')).toEqual([
      'gemma-4-E2B-it.litertlm',
      'gemma-4-E4B-it.litertlm',
    ]);
    expect(names('flagship', 'ios')).toEqual([]);
  });

  it('materializes them as downloadable Hugging Face presets', () => {
    const savedOS = Platform.OS;
    Platform.OS = 'android';
    try {
      const rules = {
        schemaVersion: '2.0.0',
        platform: 'android',
        rulesVersion: 'test',
        classifier: {
          ramBands: [{id: 'all', maxBytes: null}],
          tierMatrix: [{ramBand: 'all', socClass: 'mid', tier: 'mid'}],
          socModelToClass: {'Tensor G3': 'mid'},
        },
        tiers: {
          low: {models: []},
          mid: {models: []},
          high: {models: []},
          flagship: {models: []},
        },
      };
      const [e2b] = modelStore.resolvePresetModels(
        rules as any,
        {ramBytes: 8e9, socModel: 'Tensor G3'} as any,
        liteRTPresetCandidates('mid', 'android'),
      );

      expect(e2b).toMatchObject({
        id: 'litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm',
        name: 'Gemma 4 E2B (LiteRT-LM)',
        origin: ModelOrigin.HF,
        size: 2588147712,
        isRulePreset: true,
        downloadUrl:
          'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
      });
      expect(e2b.supportsMultimodal).toBeFalsy();
    } finally {
      Platform.OS = savedOS;
    }
  });
});
