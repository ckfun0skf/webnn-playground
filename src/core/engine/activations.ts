import type { ActivationFunction } from "../model/types";

export function applyActivation(
  values: number[],
  activation: ActivationFunction,
): number[] {
  switch (activation) {
    case "linear":
      return values;
    case "relu":
      return values.map((value) => Math.max(0, value));
    case "tanh":
      return values.map((value) => Math.tanh(value));
    case "sigmoid":
      return values.map((value) => {
        if (value >= 0) {
          return 1 / (1 + Math.exp(-value));
        }
        const exp = Math.exp(value);
        return exp / (1 + exp);
      });
    case "softmax": {
      const max = Math.max(...values);
      const exponents = values.map((value) => Math.exp(value - max));
      const total = exponents.reduce((sum, value) => sum + value, 0);
      return exponents.map((value) => value / total);
    }
  }
}
