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
      {
        key: "maxSteps",
        label: "Maximum steps",
        type: "number",
        default: 500,
        min: 1,
        max: 10000,
        step: 100,
        description: "Maximum simulation steps.",
      },
    ],
    source: snakeSource,
    network: {
      name: "Snake agent",
      inputSize: 11,
      layers: [
        { units: 16, activation: "relu" as const },
        { units: 3, activation: "softmax" as const },
      ],
    },
  },
];
