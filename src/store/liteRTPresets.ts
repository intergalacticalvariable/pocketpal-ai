import type {RuleCandidate, Tier} from '../services/deviceRules/types';

/**
 * Gemma 4 edge models packaged for LiteRT-LM by Google (litert-community,
 * Apache-2.0, not gated). The generic `.litertlm` bundle runs on the CPU and
 * GPU backends; the per-SoC NPU and `-web` variants in the same repos do not
 * load on this engine.
 *
 * The device rules are GGUF-only, so these sit beside them and join the
 * tier's candidates on Android, the only platform with a LiteRT-LM engine.
 */
const GEMMA_4_E2B: RuleCandidate = {
  model: 'gemma-4-e2b-litertlm',
  displayName: 'Gemma 4 E2B (LiteRT-LM)',
  hfRepo: 'litert-community/gemma-4-E2B-it-litert-lm',
  hfFilename: 'gemma-4-E2B-it.litertlm',
  sizeBytes: 2588147712,
};

const GEMMA_4_E4B: RuleCandidate = {
  model: 'gemma-4-e4b-litertlm',
  displayName: 'Gemma 4 E4B (LiteRT-LM)',
  hfRepo: 'litert-community/gemma-4-E4B-it-litert-lm',
  hfFilename: 'gemma-4-E4B-it.litertlm',
  sizeBytes: 3659530240,
};

// E2B needs ~1.7 GB on CPU, so it is left off the 4-6 GiB tier; E4B is a
// 3.7 GB bundle and is offered from the high tier up.
const BY_TIER: Record<Tier, RuleCandidate[]> = {
  low: [],
  mid: [GEMMA_4_E2B],
  high: [GEMMA_4_E2B, GEMMA_4_E4B],
  flagship: [GEMMA_4_E2B, GEMMA_4_E4B],
};

export const liteRTPresetCandidates = (
  tier: Tier,
  platform: string,
): RuleCandidate[] => (platform === 'android' ? BY_TIER[tier] : []);
