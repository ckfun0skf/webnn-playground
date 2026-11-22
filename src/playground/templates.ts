import blankSource from "./examples/blank.environment.js?raw";
import snakeSource from "./examples/snake.environment.js?raw";

export const PLAYGROUND_TEMPLATES = [
  {
    id: "blank",
    name: "Blank Playground",
    inputSize: 3,
    outputSize: 2,
    parameters: [],
    source: blankSource,
  },
  {
    id: "snake",
    name: "Snake Simulation",
    inputSize: 11,
    outputSize: 3,
    parameters: [
      { key: "maxStepWithoutApple", label: "Maximum steps without an apple", type: "number" as const, default: 144, min: 1, max: 10000, step: 1, description: "End the session after this many consecutive steps without eating an apple." },
      { key: "randomSeed", label: "Random seed", type: "randomSeed" as const, default: 42, description: "Shared seed across candidates within one generation." },
    ],
    source: snakeSource,
  },
];
