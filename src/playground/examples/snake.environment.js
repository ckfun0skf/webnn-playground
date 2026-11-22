export const playgroundDefinition = {
  id: "snake",
  name: "Snake",
  version: 1,
  inputSize: 11,
  outputSize: 3,
  parameters: [
    { key: "maxStepWithoutApple", label: "Maximum steps without an apple", type: "number", default: 144, min: 1, max: 10000, step: 1, description: "End the session after this many consecutive steps without eating an apple." },
    { key: "randomSeed", label: "Random seed", type: "randomSeed", default: 42, description: "Shared environment seed. Training uses the same resolved seed for every candidate in a generation." },
  ],
};

const WIDTH = 12;
const HEIGHT = 12;
const DIRECTIONS = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];

function sameCell(a, b) {
  return a.x === b.x && a.y === b.y;
}

function randomFood(ctx, snake) {
  const empty = [];
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const cell = { x, y };
      if (!snake.some((segment) => sameCell(segment, cell))) empty.push(cell);
    }
  }
  if (empty.length === 0) return null;
  return empty[Math.floor(ctx.random() * empty.length)];
}

function nextDirection(direction, action) {
  if (action === 1) return (direction + 1) % 4;
  if (action === 2) return (direction + 3) % 4;
  return direction;
}

function isDangerous(head, direction, snake) {
  const vector = DIRECTIONS[direction];
  const next = { x: head.x + vector.x, y: head.y + vector.y };
  if (next.x < 0 || next.x >= WIDTH || next.y < 0 || next.y >= HEIGHT) return 1;
  return snake.some((segment) => sameCell(segment, next)) ? 1 : 0;
}

function observe(snake, direction, food) {
  const head = snake[0];
  const right = (direction + 1) % 4;
  const left = (direction + 3) % 4;
  return [
    isDangerous(head, direction, snake),
    isDangerous(head, right, snake),
    isDangerous(head, left, snake),
    ...DIRECTIONS.map((_, index) => index === direction ? 1 : 0),
    food.x < head.x ? 1 : 0,
    food.x > head.x ? 1 : 0,
    food.y < head.y ? 1 : 0,
    food.y > head.y ? 1 : 0,
  ];
}

function argmax(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) bestIndex = index;
  }
  return bestIndex;
}

function render(ctx, snake, food, score, steps, direction, terminal) {
  const cells = snake.map((segment, index) => ({
    x: segment.x,
    y: segment.y,
    state: index === 0 ? "active" : "success",
    label: index === 0 ? "●" : "",
  }));
  if (food) cells.push({ x: food.x, y: food.y, state: "target", label: "★" });
  ctx.render({
    type: "grid",
    width: WIDTH,
    height: HEIGHT,
    cells,
    header: ["Snake simulation"],
    footer: ["Score: " + score, "Steps: " + steps, terminal ? "Finished" : "Running"],
  });
}

export async function sessionStart(ctx) {
  let snake = [{ x: 6, y: 6 }, { x: 5, y: 6 }, { x: 4, y: 6 }];
  let direction = 1;
  let food = randomFood(ctx, snake);
  let score = 0;
  let steps = 0;
  let stepsSinceApple = 0;
  const maxStepWithoutApple = Math.max(1, Number(ctx.config.maxStepWithoutApple) || 144);
  let terminal = false;

  render(ctx, snake, food, score, steps, direction, terminal);

  while (!terminal && !ctx.signal.aborted) {
    const actionValues = await ctx.network.predict(observe(snake, direction, food));
    direction = nextDirection(direction, argmax(actionValues));
    const vector = DIRECTIONS[direction];
    const nextHead = { x: snake[0].x + vector.x, y: snake[0].y + vector.y };
    const eating = sameCell(nextHead, food);
    const bodyToCheck = eating ? snake : snake.slice(0, -1);
    const hitWall = nextHead.x < 0 || nextHead.x >= WIDTH || nextHead.y < 0 || nextHead.y >= HEIGHT;
    const hitBody = bodyToCheck.some((segment) => sameCell(segment, nextHead));

    steps += 1;
    stepsSinceApple += 1;
    if (hitWall || hitBody) {
      terminal = true;
    } else {
      snake = [nextHead, ...snake];
      if (eating) {
        score += 1;
        stepsSinceApple = 0;
        food = randomFood(ctx, snake);
        if (!food) terminal = true;
      } else {
        snake.pop();
      }
    }

    if (!terminal && stepsSinceApple > maxStepWithoutApple) terminal = true;
    ctx.report({ score, steps, stepsSinceApple, terminal });
    render(ctx, snake, food, score, steps, direction, terminal);
    if (terminal || !(await ctx.step())) break;
  }

  if (ctx.signal.aborted) return;
  // Fitness is exactly the number of apples eaten. Survival time never adds points.
  await ctx.finish({ fitness: score, metrics: { score, steps, stepsSinceApple, terminal } });
}
