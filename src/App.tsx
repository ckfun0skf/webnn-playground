import { useEffect, useMemo, useRef, useState } from "react";
import { appendDenseLayer, createMLP, removeLastDenseLayer, resizeLayerUnits, updateLayerParameters, type MLPNetwork } from "./core/model";
import { compileMLP } from "./core/engine";
import { runPlaygroundSession } from "./playground/runtime";
import { DEFAULT_NEUROEVOLUTION_CONFIG, runNeuroevolution } from "./training/neuroevolution";
import type { CandidateEvaluation, GenerationSummary, NeuroevolutionConfig } from "./training/neuroevolution";
import snakeEnvironmentSource from "./playground/examples/snake.environment.js?raw";
import { deleteSavedNetwork, listSavedNetworks, listSavedPlaygrounds, saveNetwork, savePlayground, saveTrainingRun, type SavedNetworkRecord, type SavedPlaygroundRecord } from "./storage";
import ResourceLibrary from "./ResourceLibrary";\nimport type { PlaygroundDefinition } from "./playground/api";
import {
  Activity,
  Save,
  BrainCircuit,
  ChevronDown,
  CircleHelp,
  Code2,
  Cpu,
  FlaskConical,
  FolderOpen,
  Gauge,
  Layers3,
  Play,
  Plus,
  Settings2,
  Sparkles,
  Workflow,
} from "lucide-react";

type Workspace = "network" | "training" | "playground" | "library" | "resources";

const navigation: { id: Workspace; label: string; icon: typeof BrainCircuit }[] = [
  { id: "network", label: "Network", icon: BrainCircuit },
  { id: "training", label: "Training", icon: Activity },
  { id: "playground", label: "Playground", icon: FlaskConical },
  { id: "library", label: "Network Library", icon: FolderOpen },
  { id: "resources", label: "Resource Library", icon: Workflow },
];

function detectCapabilities() {
  const secure = window.isSecureContext;
  const webnn = "ml" in navigator;
  return { secure, webnn };
}

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>("network");
  const [projectName] = useState("Untitled experiment");
  const [network, setNetwork] = useState<MLPNetwork | null>(null);
  const [environmentSource, setEnvironmentSource] = useState(DEFAULT_ENVIRONMENT_CODE);
  const [libraryRevision, setLibraryRevision] = useState(0);
  const [resourceRevision, setResourceRevision] = useState(0);
  const [inputText, setInputText] = useState("0, 0, 0");
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const capabilities = useMemo(detectCapabilities, []);
  const saveToLibrary = async (model: MLPNetwork) => {
    await saveNetwork(model);
    setLibraryRevision((revision) => revision + 1);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="WebNN Playground home">
          <span className="brand-mark"><BrainCircuit size={19} /></span>
          <span>webnn<span className="brand-muted">playground</span></span>
        </a>
        <div className="project-switcher">
          <span className="project-dot" />
          <span>{projectName}</span>
          <ChevronDown size={14} />
        </div>
        <div className="topbar-spacer" />
        <div className="runtime-status">
          <span className={capabilities.secure ? "status-dot good" : "status-dot warn"} />
          <span>{capabilities.secure ? "Secure context" : "Insecure context"}</span>
        </div>
        <div className="runtime-status">
          <span className={capabilities.webnn ? "status-dot good" : "status-dot neutral"} />
          <span>WebNN {capabilities.webnn ? "detected" : "not detected"}</span>
        </div>
        <button className="icon-button" title="Settings" aria-label="Settings"><Settings2 size={17} /></button>
        <button className="icon-button" title="Help" aria-label="Help"><CircleHelp size={17} /></button>
      </header>

      <aside className="sidebar">
        <div className="sidebar-section-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main workspace">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              className={`nav-item ${workspace === id ? "active" : ""}`}
              key={id}
              onClick={() => setWorkspace(id)}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
              {id === "playground" && <span className="nav-new">CORE</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-divider" />
        <div className="sidebar-section-label">PROJECT</div>
        <button className="project-tree-item selected"><Layers3 size={16} /> Neural network</button>
        <button className="project-tree-item"><Code2 size={16} /> Environment code</button>
        <button className="project-tree-item"><Workflow size={16} /> Training runs</button>
        <div className="sidebar-bottom">
          <div className="storage-card">
            <div className="storage-title"><span>LOCAL WORKSPACE</span><span className="storage-indicator" /></div>
            <p>Your experiments stay in this browser.</p>
            <button className="text-button"><Plus size={13} /> New project</button>
          </div>
          <div className="sidebar-footer"><span>WEBNN PLAYGROUND</span><span>v0.1.0 · FOUNDATION</span></div>
        </div>
      </aside>

      <main className="main-content">
        <div className="page-heading">
          <div>
            <div className="eyebrow">EXPERIMENT WORKSPACE <span>/</span> {workspace.toUpperCase()}</div>
            <h1>{navigation.find((item) => item.id === workspace)?.label}</h1>
            <p className="page-subtitle">{getSubtitle(workspace)}</p>
          </div>
          <div className="heading-actions">
            <button className="secondary-button"><FolderOpen size={15} /> Open project</button>
            <button className="primary-button"><Plus size={15} /> New {workspace === "playground" ? "environment" : workspace === "network" ? "network" : "experiment"}</button>
          </div>
        </div>
        {workspace === "network" && !network && (
          <section className="getting-started panel">
            <div className="getting-started-copy">
              <span className="panel-kicker">NEW HERE? START WITH A WORKING EXAMPLE</span>
              <h2>Train a neural network to play Snake</h2>
              <p>No setup required. Load the example, watch the game board, then evolve a better Snake agent.</p>
              <div className="getting-started-steps">
                <span><b>1</b> Load Snake</span><span><b>2</b> Run the simulation</span><span><b>3</b> Train the agent</span>
              </div>
            </div>
            <button className="primary-button" onClick={() => {
              setEnvironmentSource(snakeEnvironmentSource);
              setNetwork(createMLP({ name: "Snake agent", inputSize: 11, layers: [{ units: 16, activation: "relu" }, { units: 3, activation: "softmax" }] }));
              setWorkspace("playground");
            }}><Play size={14} /> Try Snake</button>
          </section>
        )}

        {workspace === "network" && <NetworkOverview network={network} setNetwork={setNetwork} onSaveNetwork={saveToLibrary} inputText={inputText} setInputText={setInputText} prediction={prediction} setPrediction={setPrediction} inferenceError={inferenceError} setInferenceError={setInferenceError} />}
        {workspace === "training" && <TrainingStudio network={network} setNetwork={setNetwork} onSaveNetwork={saveToLibrary} resourceRevision={resourceRevision} setWorkspace={setWorkspace} />}
        {workspace === "playground" && <PlaygroundOverview network={network} setNetwork={setNetwork} source={environmentSource} setSource={setEnvironmentSource} onPlaygroundSaved={() => setResourceRevision((v) => v + 1)} resourceRevision={resourceRevision} />}
        {workspace === "library" && <NetworkLibrary revision={libraryRevision} onLoad={(model) => { setNetwork(model); setWorkspace("network"); }} />}
        {workspace === "resources" && <ResourceLibrary source={environmentSource} onLoadPlayground={(source) => { setEnvironmentSource(source); setWorkspace("playground"); }} onLoadNetwork={(model) => { setNetwork(model); setWorkspace("network"); }} revision={resourceRevision} onChanged={() => setResourceRevision((v) => v + 1)} />}
      </main>
    </div>
  );
}

function getSubtitle(workspace: Workspace) {
  switch (workspace) {
    case "network": return "Design a model, inspect its structure, and run inference.";
    case "training": return "Configure learning algorithms and track model improvement.";
    case "playground": return "Build simulations in code and connect them to neural networks.";
    case "library": return "Keep trained networks, checkpoints, and reusable models.";
    case "resources": return "Manage reusable Playgrounds and saved Neuroevolution training runs.";
  }
}

function NetworkOverview({
  network,
  setNetwork,
  onSaveNetwork,
  inputText,
  setInputText,
  prediction,
  setPrediction,
  inferenceError,
  setInferenceError,
}: {
  network: MLPNetwork | null;
  setNetwork: (network: MLPNetwork | null) => void;
  onSaveNetwork: (network: MLPNetwork) => Promise<void>;
  inputText: string;
  setInputText: (value: string) => void;
  prediction: number[] | null;
  setPrediction: (value: number[] | null) => void;
  inferenceError: string | null;
  setInferenceError: (value: string | null) => void;
}) {
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const createNetwork = () => {
    const next = createMLP({
      name: "My first MLP",
      inputSize: 3,
      layers: [{ units: 4, activation: "relu" }, { units: 2, activation: "softmax" }],
    });
    setNetwork(next);
    setInputText("0, 0, 0");
    setPrediction(null);
    setInferenceError(null);
  };

  const runInference = () => {
    if (!network) return;
    try {
      const values = inputText.split(",").map((part) => Number(part.trim()));
      setPrediction(compileMLP(network)(values));
      setInferenceError(null);
    } catch (error) {
      setPrediction(null);
      setInferenceError(error instanceof Error ? error.message : "Inference failed.");
    }
  };

  return (
    <div className="content-grid">
      <section className="panel network-panel">
        <div className="panel-heading">
          <div><div className="panel-kicker">MODEL CANVAS</div><h2>{network ? network.name : "Network architecture"}</h2></div>
          <div className="panel-tools"><span className="model-chip">MLP</span><button className="small-icon-button" title="Model options" aria-label="Model options"><Settings2 size={15} /></button></div>
        </div>
        <div className="empty-canvas">
          <div className="canvas-grid" />
          {network ? (
            <div className="live-network">
              <div className="layer-column">
                <span className="layer-label">INPUT · {network.inputSize}</span>
                {Array.from({ length: network.inputSize }, (_, n) => <span className="neuron input-neuron" key={n}>x{n + 1}</span>)}
              </div>
              {network.layers.map((layer, layerIndex) => (
                <div className="live-layer-group" key={layer.id}>
                  <div className="connection-lines"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
                  <div className="layer-column">
                    <span className="layer-label">{layerIndex === network.layers.length - 1 ? "OUTPUT" : "HIDDEN"} · {layer.units}</span>
                    {Array.from({ length: layer.units }, (_, n) => <span className={`neuron ${layerIndex === network.layers.length - 1 ? "output-neuron" : "hidden-neuron"}`} key={n}>{layerIndex === network.layers.length - 1 ? `y${n + 1}` : `h${n + 1}`}</span>)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="network-placeholder">
              <div className="layer-column"><span className="layer-label">INPUT</span>{[0, 1, 2].map((n) => <span className="neuron input-neuron" key={n}>{["x₁", "x₂", "x₃"][n]}</span>)}</div>
              <div className="connection-lines"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
              <div className="layer-column"><span className="layer-label">HIDDEN</span>{[0, 1, 2, 3].map((n) => <span className="neuron hidden-neuron" key={n}>h{n + 1}</span>)}</div>
              <div className="connection-lines"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
              <div className="layer-column"><span className="layer-label">OUTPUT</span>{[0, 1].map((n) => <span className="neuron output-neuron" key={n}>y{n + 1}</span>)}</div>
            </div>
          )}
          {!network && <div className="canvas-empty-caption"><Sparkles size={13} /> Create a network to inspect its live architecture</div>}
        </div>
        <div className="canvas-toolbar">
          <span><span className="legend-dot input" /> Input <span className="legend-dot hidden" /> Hidden <span className="legend-dot output" /> Output</span>
          <span>{network ? `${network.inputSize} → ${network.layers.map((layer) => layer.units).join(" → ")}` : "Canvas · 100%"}</span>
        </div>
        {network && (
          <div className="model-details">
            <div className="model-detail-heading"><div><strong>Layer parameters</strong><span>{network.layers.length} dense layers</span></div><button className="secondary-button" onClick={() => { setNetwork(appendDenseLayer(network, 4)); setPrediction(null); }}><Plus size={13} /> Add layer</button></div>
            {network.layers.map((layer, index) => (
              <div className="layer-editor" key={layer.id}>
                <div className="layer-detail">
                  <div className="layer-heading-controls"><strong>Layer {index + 1}</strong><label className="unit-count-control">UNITS<input type="number" min="1" max="128" value={layer.units} onChange={(event) => { if (Number.isInteger(event.target.valueAsNumber) && event.target.valueAsNumber > 0) { setNetwork(resizeLayerUnits(network, layer.id, event.target.valueAsNumber)); setPrediction(null); } }} /></label>{index === network.layers.length - 1 && network.layers.length > 1 && <button className="text-button remove-layer-button" onClick={() => { setNetwork(removeLastDenseLayer(network)); setPrediction(null); }}>Remove layer</button>}</div>
                  <label className="activation-control">ACTIVATION
                    <select
                      value={layer.activation}
                      onChange={(event) => {
                        setNetwork(updateLayerParameters(network, layer.id, { activation: event.target.value as typeof layer.activation }));
                        setPrediction(null);
                      }}
                    >
                      {["linear", "relu", "tanh", "sigmoid", "softmax"].map((activation) => <option key={activation} value={activation}>{activation}</option>)}
                    </select>
                  </label>
                </div>
                {layer.weights.map((row, neuronIndex) => (
                  <div className="parameter-row" key={neuronIndex}>
                    <span className="parameter-neuron">N{neuronIndex + 1}</span>
                    <div className="parameter-values">
                      {row.map((weight, inputIndex) => (
                        <label key={inputIndex}>W{inputIndex + 1}
                          <input
                            type="number"
                            step="0.01"
                            value={weight}
                            onChange={(event) => {
                              const nextWeights = layer.weights.map((weightsRow) => [...weightsRow]);
                              nextWeights[neuronIndex][inputIndex] = event.target.valueAsNumber;
                              if (Number.isFinite(event.target.valueAsNumber)) {
                                setNetwork(updateLayerParameters(network, layer.id, { weights: nextWeights }));
                                setPrediction(null);
                              }
                            }}
                          />
                        </label>
                      ))}
                    </div>
                    <label className="parameter-bias">BIAS
                      <input
                        type="number"
                        step="0.01"
                        value={layer.biases[neuronIndex]}
                        onChange={(event) => {
                          if (!Number.isFinite(event.target.valueAsNumber)) return;
                          const nextBiases = [...layer.biases];
                          nextBiases[neuronIndex] = event.target.valueAsNumber;
                          setNetwork(updateLayerParameters(network, layer.id, { biases: nextBiases }));
                          setPrediction(null);
                        }}
                      />
                    </label>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </section>

      <aside className="right-column">
        <section className="panel quick-panel">
          <div className="panel-kicker">{network ? "INFERENCE" : "GET STARTED"}</div>
          <h2>{network ? "Run your network" : "Create your first network"}</h2>
          <p>{network ? "Enter one value per input neuron, separated by commas, then execute forward propagation." : "Start with a multilayer perceptron. The network will use the same model format as training and Playground environments."}</p>
          {!network ? (
            <button className="primary-button full-button" onClick={createNetwork}><Plus size={15} /> Create MLP network</button>
          ) : (
            <>
              <label className="field-label" htmlFor="network-input">INPUT VALUES</label>
              <input id="network-input" className="network-input" value={inputText} onChange={(event) => setInputText(event.target.value)} />
              <button className="primary-button full-button" onClick={runInference}><Play size={14} /> Run inference</button>
              {inferenceError && <p className="inference-error">{inferenceError}</p>}
              {prediction && <div className="prediction-result"><span>OUTPUT</span><strong>{prediction.map((value) => value.toFixed(5)).join(" · ")}</strong></div>}
              <button className="secondary-button full-button" onClick={async () => {
                if (!network) return;
                try { await onSaveNetwork(network); setSaveMessage("Saved to Network Library."); }
                catch (error) { setSaveMessage(error instanceof Error ? error.message : String(error)); }
              }}><FolderOpen size={14} /> Save to Network Library</button>
              {saveMessage && <p className="save-message">{saveMessage}</p>}
              <button className="secondary-button full-button" onClick={() => { setNetwork(null); setPrediction(null); setInferenceError(null); }}>Close network</button>
            </>
          )}
          <button className="secondary-button full-button" onClick={createNetwork}><Code2 size={15} /> {network ? "Reset to a new MLP" : "Create example MLP"}</button>
        </section>
        <section className="panel backend-panel">
          <div className="panel-heading compact"><div className="panel-kicker">EXECUTION BACKENDS</div><Gauge size={16} /></div>
          <div className="backend-row"><span className="backend-icon js">JS</span><div><strong>JavaScript</strong><small>Active · MLP forward pass</small></div><span className="backend-state available">READY</span></div>
          <div className="backend-row"><span className="backend-icon webnn"><Cpu size={16} /></span><div><strong>WebNN</strong><small>Optional · browser dependent</small></div><span className={`backend-state ${"ml" in navigator ? "available" : "unavailable"}`}>{ "ml" in navigator ? "DETECTED" : "UNAVAILABLE"}</span></div>
        </section>
      </aside>
    </div>
  );
}

const DEFAULT_ENVIRONMENT_CODE = `export async function sessionStart(ctx) {
  const inputSize = Number(ctx.config.inputSize);
  const outputSize = Number(ctx.config.outputSize);
  let totalError = 0;

  // This example evaluates the network on three generated observations.
  for (let step = 0; step < 3; step += 1) {
    if (ctx.signal.aborted) return;

    const input = Array.from(
      { length: inputSize },
      (_, index) => index === step % inputSize ? 1 : 0
    );
    const target = Array.from(
      { length: outputSize },
      (_, index) => index === step % outputSize ? 1 : 0
    );

    const output = await ctx.network.predict(input);
    const error = output.reduce(
      (sum, value, index) => sum + (value - target[index]) ** 2,
      0
    );

    totalError += error;
    ctx.report({ step: step + 1, error, totalError });
    ctx.render({ type: "observation", step: step + 1, input, output, target, error });
    await ctx.yield();
  }

  await ctx.finish({
    fitness: 1 / (1 + totalError),
    metrics: { totalError, observations: 3 }
  });
}

export async function sessionEnd(ctx, result) {
  // Optional cleanup after a successful evaluation.
}`;

type GridVisualization = {
  type: "grid";
  width: number;
  height: number;
  cells?: { x: number; y: number; state?: string; label?: string | number }[];
  title?: string;
  metrics?: { label: string; value: string | number }[];
};

type MetricsVisualization = {
  type: "metrics";
  title?: string;
  metrics: { label: string; value: string | number; unit?: string }[];
};

type SnakeVisualization = {
  type: "snake";
  width: number;
  height: number;
  snake: { x: number; y: number; head?: boolean }[];
  food: { x: number; y: number } | null;
  score: number;
  steps: number;
  terminal: boolean;
};

function EnvironmentVisualization({ view }: { view: unknown }) {
  if (view && typeof view === "object" && (view as SnakeVisualization).type === "snake") {
    const snakeView = view as SnakeVisualization;
    return (
      <div className="snake-visualization">
        <div className="snake-board" style={{ gridTemplateColumns: "repeat(" + snakeView.width + ", minmax(0, 1fr))" }}>
          {Array.from({ length: snakeView.width * snakeView.height }, (_, index) => {
            const x = index % snakeView.width;
            const y = Math.floor(index / snakeView.width);
            const segment = snakeView.snake.find((part) => part.x === x && part.y === y);
            const isFood = snakeView.food?.x === x && snakeView.food?.y === y;
            const cellClass = segment ? (segment.head ? "head" : "body") : isFood ? "food" : "";
            return <div key={index} className={"snake-board-cell " + cellClass} />;
          })}
        </div>
        <div className="snake-board-stats"><span>Score <strong>{snakeView.score}</strong></span><span>Steps <strong>{snakeView.steps}</strong></span><span>{snakeView.terminal ? "Finished" : "Running"}</span></div>
      </div>
    );
  }

  if (view && typeof view === "object" && (view as GridVisualization).type === "grid") {
    const grid = view as GridVisualization;
    const width = Math.max(1, Math.min(64, Math.floor(grid.width || 1)));
    const height = Math.max(1, Math.min(64, Math.floor(grid.height || 1)));
    const cells = Array.isArray(grid.cells) ? grid.cells : [];
    return <div className="generic-playground-view">
      {grid.title && <strong className="generic-view-title">{grid.title}</strong>}
      <div className="generic-playground-grid" style={{ gridTemplateColumns: "repeat(" + width + ", minmax(0, 1fr))" }}>
        {Array.from({ length: width * height }, (_, index) => {
          const x = index % width, y = Math.floor(index / width);
          const cell = cells.find((item) => item.x === x && item.y === y);
          const state = ["active", "target", "wall", "success", "danger", "muted"].includes(cell?.state ?? "") ? cell!.state : "empty";
          return <div key={index} className={"generic-playground-cell " + state} title={cell?.label === undefined ? undefined : String(cell.label)}>{cell?.label ?? ""}</div>;
        })}
      </div>
      {grid.metrics && <div className="generic-view-metrics">{grid.metrics.map((item) => <span key={item.label}>{item.label} <strong>{item.value}</strong></span>)}</div>}
    </div>;
  }
  if (view && typeof view === "object" && (view as MetricsVisualization).type === "metrics") {
    const data = view as MetricsVisualization;
    return <div className="generic-playground-view">
      {data.title && <strong className="generic-view-title">{data.title}</strong>}
      <div className="generic-view-metrics">{data.metrics.map((item) => <span key={item.label}>{item.label} <strong>{item.value}{item.unit ? " " + item.unit : ""}</strong></span>)}</div>
    </div>;
  }
  return <pre>{view === null ? "Environment-defined state will appear here." : JSON.stringify(view, null, 2)}</pre>;
}
function TrainingStudio({
  network,
  setNetwork,
  onSaveNetwork,
  setWorkspace,
  resourceRevision,
}: {
  network: MLPNetwork | null;
  setNetwork: (network: MLPNetwork | null) => void;
  onSaveNetwork: (network: MLPNetwork) => Promise<void>;
  resourceRevision: number;
  setWorkspace: (workspace: Workspace) => void;
}) {
  const [config, setConfig] = useState<NeuroevolutionConfig>({ ...DEFAULT_NEUROEVOLUTION_CONFIG });
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "failed" | "cancelled">("idle");
  const [best, setBest] = useState<CandidateEvaluation | null>(null);
  const [history, setHistory] = useState<GenerationSummary[]>([]);
  const [events, setEvents] = useState<{ generation: number; candidateIndex: number; fitness: number; error?: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [playbackDelay, setPlaybackDelay] = useState(80);
  const [populationViews, setPopulationViews] = useState<Record<number, { view: unknown; fitness?: number; finished?: boolean; error?: string }>>({});
  const [populationReplays, setPopulationReplays] = useState<Record<string, unknown[]>>({});
  const [populationColumns, setPopulationColumns] = useState(4);
  const [selectedAgent, setSelectedAgent] = useState<{ generation: number; index: number } | null>(null);
  const [replayFrameIndex, setReplayFrameIndex] = useState(0);
  const [isReplaying, setIsReplaying] = useState(false);
  const [environmentMaxSteps, setEnvironmentMaxSteps] = useState(500);
  const [activeGeneration, setActiveGeneration] = useState(0);
  const [savedPlaygrounds, setSavedPlaygrounds] = useState<SavedPlaygroundRecord[]>([]);
  const [selectedPlaygroundId, setSelectedPlaygroundId] = useState("");
  const [savedCompatibleNetworks, setSavedCompatibleNetworks] = useState<SavedNetworkRecord[]>([]);
  const [selectedNetworkId, setSelectedNetworkId] = useState("");
  const selectedPlayground = savedPlaygrounds.find((item) => item.id === selectedPlaygroundId) ?? null;
  useEffect(() => {
    let active = true;
    Promise.all([listSavedPlaygrounds(), listSavedNetworks()]).then(([playgrounds, networks]) => {
      if (!active) return;
      setSavedPlaygrounds(playgrounds);
      setSavedCompatibleNetworks(networks);
      setSelectedPlaygroundId((current) => current || playgrounds[0]?.id || "");
    }).catch((loadError) => setError(loadError instanceof Error ? loadError.message : String(loadError)));
    return () => { active = false; };
  }, [resourceRevision]);
  const compatibleNetworks = selectedPlayground
    ? savedCompatibleNetworks.filter((item) => item.network.inputSize === selectedPlayground.definition.inputSize && item.network.layers[item.network.layers.length - 1]?.units === selectedPlayground.definition.outputSize)
    : [];
  const [environmentParameters, setEnvironmentParameters] = useState<Record<string, number | boolean | string>>({});
  useEffect(() => {
    if (!selectedPlayground) { setEnvironmentParameters({}); return; }
    setEnvironmentParameters(Object.fromEntries(selectedPlayground.definition.parameters.map((parameter) => [parameter.key, parameter.default])));
  }, [selectedPlaygroundId]);

  useEffect(() => {
    if (!isReplaying || !selectedAgent) return;
    const frames = populationReplays[selectedAgent.generation + "-" + selectedAgent.index] ?? [];
    if (replayFrameIndex >= frames.length - 1) {
      setIsReplaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      setReplayFrameIndex((index) => Math.min(index + 1, frames.length - 1));
    }, Math.max(30, playbackDelay || 30));
    return () => window.clearInterval(timer);
  }, [isReplaying, selectedAgent, replayFrameIndex, playbackDelay, populationReplays]);
  const abortRef = useRef<AbortController | null>(null);

  type NumericConfigKey = "populationSize" | "generations" | "eliteCount" | "environmentSeed" | "tournamentSize" | "crossoverRate" | "mutationRate" | "mutationScale" | "evaluationTimeoutMs" | "seed";
  const updateNumber = (key: NumericConfigKey, value: number) => {
    if (!Number.isFinite(value)) return;
    setConfig((current) => {
      if (key === "populationSize") {
        return { ...current, populationSize: value, eliteCount: Math.min(current.eliteCount, Math.max(1, value - 1)) };
      }
      return { ...current, [key]: value } as NeuroevolutionConfig;
    });
  };

  const startTraining = async () => {
    if (!selectedPlayground) { setError("Save a Playground in Resource Library, then select it here."); return; }
    const networkIsCompatible = network && network.inputSize === selectedPlayground.definition.inputSize && network.layers[network.layers.length - 1]?.units === selectedPlayground.definition.outputSize;
    const activeNetwork = networkIsCompatible ? network! : createMLP({
      name: selectedPlayground.name + " seed network",
      inputSize: selectedPlayground.definition.inputSize,
      layers: [{ units: Math.max(4, selectedPlayground.definition.outputSize * 2), activation: "relu" }, { units: selectedPlayground.definition.outputSize, activation: "softmax" }],
    });
    if (!networkIsCompatible) setNetwork(activeNetwork);

    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("running");
    setBest(null);
    setHistory([]);
    setEvents([]);
    setPopulationViews({});
    setPopulationReplays({});
    setSelectedAgent(null);
    setIsReplaying(false);
    setActiveGeneration(0);
    setError(null);

    try {
      const outputSize = activeNetwork.layers[activeNetwork.layers.length - 1].units;
      const result = await runNeuroevolution({
        initialNetwork: activeNetwork,
        environmentSource: selectedPlayground.source,
        config,
        configForEnvironment: { inputSize: activeNetwork.inputSize, outputSize, stepDelayMs: playbackDelay, maxSteps: environmentMaxSteps, ...environmentParameters },
        signal: controller.signal,
        onCandidate: (candidate) => {
          setEvents((current) => [...current, {
            generation: candidate.generation,
            candidateIndex: candidate.candidateIndex,
            fitness: candidate.fitness,
            error: candidate.error,
          }]);
          setBest((current) => !current || candidate.fitness > current.fitness ? candidate : current);
          setPopulationViews((current) => ({
            ...current,
            [candidate.candidateIndex]: {
              ...(current[candidate.candidateIndex] ?? { view: null }),
              fitness: candidate.fitness,
              finished: true,
              error: candidate.error,
            },
          }));
        },
        onPopulationRender: (generation, candidateIndex, view) => {
          setActiveGeneration(generation);
          setPopulationViews((current) => ({ ...current, [candidateIndex]: { ...current[candidateIndex], view } }));
          const replayKey = generation + "-" + candidateIndex;
          setPopulationReplays((current) => ({
            ...current,
            [replayKey]: [...(current[replayKey] ?? []), view].slice(-1200),
          }));
        },
        onGeneration: (summary) => {
          setHistory((current) => [...current, summary]);
          if (summary.generation < config.generations - 1) setPopulationViews({});
        },
      });
      setBest(result.best);
      const bestNetwork = { ...structuredClone(result.best.network), id: crypto.randomUUID(), name: selectedPlayground.name + " best · " + new Date().toLocaleDateString() };
      await onSaveNetwork(bestNetwork);
      await saveTrainingRun({
        id: "training-" + crypto.randomUUID(), name: selectedPlayground.name + " · " + new Date().toLocaleString(), status: "completed",
        playgroundId: selectedPlayground.id, playgroundName: selectedPlayground.name, playgroundVersion: selectedPlayground.version,
        playgroundSource: selectedPlayground.source, initialNetwork: structuredClone(activeNetwork), evolutionConfig: structuredClone(config) as unknown as Record<string, unknown>,
        environmentConfig: { maxSteps: environmentMaxSteps, stepDelayMs: playbackDelay, seed: config.environmentSeed, ...environmentParameters },
        bestNetwork, bestFitness: result.best.fitness, generationHistory: structuredClone(result.history) as unknown as Record<string, unknown>[],
        createdAt: Date.now(), completedAt: Date.now(),
      });
      setSaveMessage("Training Run saved; best candidate added as a new Network.");
      setStatus("completed");
    } catch (trainingError) {
      setError(trainingError instanceof Error ? trainingError.message : String(trainingError));
      setStatus(controller.signal.aborted ? "cancelled" : "failed");
    } finally {
      abortRef.current = null;
    }
  };

  const cancelTraining = () => abortRef.current?.abort("Training cancelled by user.");

  return (
    <div className="training-studio">
      <section className="panel training-config-panel">
        <div className="panel-heading">
          <div><div className="panel-kicker">TRAINING ALGORITHM</div><h2>Neuroevolution</h2></div>
          <span className="model-chip">GENETIC ALGORITHM</span>
        </div>
        <div className="training-config-body">
          <p className="training-description">Evaluate candidate networks inside your JavaScript Playground. Fitness is calculated by the environment; the evolution engine handles selection, crossover, mutation, and generations.</p>
          <div className="training-config-grid">
            <label>Playground<select value={selectedPlaygroundId} onChange={(event) => { setSelectedPlaygroundId(event.target.value); setNetwork(null); setSelectedNetworkId(""); }}><option value="">Select a saved Playground…</option>{savedPlaygrounds.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.definition.inputSize}→{p.definition.outputSize}</option>)}</select></label>
            <label>Compatible Network<select value={selectedNetworkId} onChange={(event) => { setSelectedNetworkId(event.target.value); const item = compatibleNetworks.find((x) => x.id === event.target.value); if (item) setNetwork(item.network); }}><option value="">Use current / generated network</option>{compatibleNetworks.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            {selectedPlayground?.definition.parameters.map((parameter) => <label key={parameter.key}>{parameter.label}
              {parameter.type === "boolean" ? <input type="checkbox" checked={Boolean(environmentParameters[parameter.key])} onChange={(event) => setEnvironmentParameters((current) => ({ ...current, [parameter.key]: event.target.checked }))} /> :
                <input type={parameter.type === "number" ? "number" : "text"} min={parameter.min} max={parameter.max} step={parameter.step} value={String(environmentParameters[parameter.key] ?? parameter.default)} onChange={(event) => setEnvironmentParameters((current) => ({ ...current, [parameter.key]: parameter.type === "number" ? event.target.valueAsNumber : event.target.value }))} />}
            </label>)}
            <label>Population size<input type="number" min="2" max="500" value={config.populationSize} onChange={(event) => updateNumber("populationSize", event.target.valueAsNumber)} /></label>
            <label>Generations<input type="number" min="1" max="1000" value={config.generations} onChange={(event) => updateNumber("generations", event.target.valueAsNumber)} /></label>
            <label>Elite count<input type="number" min="1" max={Math.max(1, config.populationSize - 1)} value={config.eliteCount} onChange={(event) => updateNumber("eliteCount", event.target.valueAsNumber)} /></label>
            <label>Tournament size<input type="number" min="1" max="100" value={config.tournamentSize} onChange={(event) => updateNumber("tournamentSize", event.target.valueAsNumber)} /></label>
            <label>Mutation rate<input type="number" min="0" max="1" step="0.01" value={config.mutationRate} onChange={(event) => updateNumber("mutationRate", event.target.valueAsNumber)} /></label>
            <label>Mutation scale<input type="number" min="0" step="0.01" value={config.mutationScale} onChange={(event) => updateNumber("mutationScale", event.target.valueAsNumber)} /></label>
            <label>Crossover rate<input type="number" min="0" max="1" step="0.01" value={config.crossoverRate} onChange={(event) => updateNumber("crossoverRate", event.target.valueAsNumber)} /></label>
            <label>Evolution seed<input type="number" step="1" value={config.seed} onChange={(event) => updateNumber("seed", event.target.valueAsNumber)} /></label>
            <label>Environment seed<input type="number" step="1" value={config.environmentSeed} onChange={(event) => updateNumber("environmentSeed", event.target.valueAsNumber)} /></label>
            <label>Max steps per agent<input type="number" min="1" max="10000" step="1" value={environmentMaxSteps} onChange={(event) => setEnvironmentMaxSteps(Math.max(1, event.target.valueAsNumber || 1))} /></label>
          </div>
          <div className="training-environment-link">
            <div><span className="panel-kicker">SELECTED PLAYGROUND</span><strong>{selectedPlayground ? selectedPlayground.name + " · " + selectedPlayground.definition.inputSize + " inputs / " + selectedPlayground.definition.outputSize + " outputs" : "Save a Playground in Resource Library first"}</strong></div>
            <button className="text-button" onClick={() => setWorkspace("playground")}><Code2 size={13} /> Edit environment</button>
          </div>
          <div className="playback-speed-control">
            <label htmlFor="training-speed">Population playback speed</label>
            <input id="training-speed" type="range" min="0" max="500" step="10" disabled={status === "running"} value={playbackDelay} onChange={(event) => setPlaybackDelay(Number(event.target.value))} />
            <span>{playbackDelay === 0 ? "Fastest" : playbackDelay + " ms / step"}</span>
          </div>
          <div className="training-actions">
            {status === "running" ? (
              <button className="secondary-button" onClick={cancelTraining}>Cancel training</button>
            ) : (
              <button className="primary-button" onClick={startTraining} disabled={!selectedPlayground}><Play size={14} /> Start evolution</button>
            )}
            <span className="training-status"><span className={`status-dot ${status === "running" || status === "completed" ? "good" : status === "failed" ? "warn" : "neutral"}`} />{status}</span>
          </div>
          {error && <div className="environment-error">{error}</div>}
        </div>
      </section>

      <section className="panel population-live-panel">
        <div className="panel-heading compact">
          <div><div className="panel-kicker">LIVE POPULATION</div><strong>Generation {activeGeneration + 1} · {config.populationSize} agents</strong></div>
          <span>{Object.values(populationViews).filter((item) => item.finished).length} / {config.populationSize} finished</span>
        </div>
        <p className="population-live-description">Every candidate is playing its own Snake game simultaneously. Each board is driven by that candidate's neural network.</p>
        <div className="population-grid-controls">
          <label>Grid density
            <select value={populationColumns} onChange={(event) => setPopulationColumns(Number(event.target.value))}>
              <option value={2}>2 columns</option><option value={3}>3 columns</option><option value={4}>4 columns</option>
              <option value={5}>5 columns</option><option value={6}>6 columns</option><option value={8}>8 columns</option>
            </select>
          </label>
        </div>
        <div className="population-board-grid" style={{ gridTemplateColumns: "repeat(" + populationColumns + ", minmax(0, 1fr))" }}>
          {Array.from({ length: config.populationSize }, (_, index) => {
            const candidate = populationViews[index];
            return <button type="button" className={"population-agent-card " + (candidate?.finished ? "is-finished" : "")} key={index}
              onClick={() => { setSelectedAgent({ generation: activeGeneration, index }); setReplayFrameIndex(0); setIsReplaying(false); }}>
              <div className="population-agent-heading"><strong>Agent {String(index + 1).padStart(2, "0")}</strong><span className={candidate?.error ? "candidate-failed" : candidate?.finished ? "candidate-ok" : "population-agent-running"}>{candidate?.error ? "FAILED" : candidate?.finished ? "DONE" : candidate?.view ? "PLAYING" : status === "running" ? "STARTING" : "WAITING"}</span></div>
              {candidate?.view ? <EnvironmentVisualization view={candidate.view} /> : <div className="population-board-placeholder">Waiting for first frame…</div>}
              {candidate?.finished && <div className="population-agent-fitness">Fitness <strong>{candidate.fitness?.toFixed(2) ?? "—"}</strong></div>}
              <span className="population-agent-open">{candidate?.finished ? "View · Replay available" : "View live agent"}</span>
            </button>;
          })}
        </div>
        {selectedAgent && (() => {
          const key = selectedAgent.generation + "-" + selectedAgent.index;
          const frames = populationReplays[key] ?? [];
          const selectedCandidate = selectedAgent.generation === activeGeneration ? populationViews[selectedAgent.index] : undefined;
          const replayView = frames[Math.min(replayFrameIndex, Math.max(0, frames.length - 1))];
          const canReplay = Boolean(!selectedCandidate?.error && (selectedCandidate?.finished || events.some((item) => item.generation === selectedAgent.generation && item.candidateIndex === selectedAgent.index && !item.error)));
          return <div className="agent-detail-backdrop" role="presentation" onClick={() => { setSelectedAgent(null); setIsReplaying(false); }}>
            <section className="panel agent-detail-modal" role="dialog" aria-modal="true" aria-label="Agent details" onClick={(event) => event.stopPropagation()}>
              <div className="panel-heading"><div><div className="panel-kicker">AGENT DETAIL</div><h2>Generation {selectedAgent.generation + 1} · Agent {selectedAgent.index + 1}</h2></div><button className="text-button" onClick={() => { setSelectedAgent(null); setIsReplaying(false); }}>Close ×</button></div>
              <div className="agent-detail-board"><EnvironmentVisualization view={isReplaying ? replayView : (selectedCandidate?.view ?? replayView)} /></div>
              <div className="agent-replay-controls">
                <span>{frames.length ? "Frame " + Math.min(replayFrameIndex + 1, frames.length) + " / " + frames.length : "No frames captured"}</span>
                <input aria-label="Replay position" type="range" min="0" max={Math.max(0, frames.length - 1)} value={Math.min(replayFrameIndex, Math.max(0, frames.length - 1))} disabled={!canReplay || frames.length < 2 || isReplaying} onChange={(event) => setReplayFrameIndex(Number(event.target.value))} />
                <button className="primary-button" disabled={!canReplay || frames.length < 2} onClick={() => { setReplayFrameIndex(0); setIsReplaying((value) => !value); }}>{isReplaying ? "Pause replay" : "Replay"}</button>
              </div>
              {!canReplay && <p className="empty-logs">Replay becomes available after this agent finishes.</p>}
            </section>
          </div>;
        })()}
      </section>
      <div className="training-results-grid">
        <section className="panel training-progress-panel">
          <div className="panel-heading compact"><div className="panel-kicker">GENERATION HISTORY</div><span>{history.length} / {config.generations}</span></div>
          {history.length === 0 ? <p className="empty-logs">Generation statistics will appear after the first population has been evaluated.</p> : (
            <div className="generation-list">
              {history.map((entry) => (
                <div className="generation-row" key={entry.generation}>
                  <strong>Gen {entry.generation + 1}</strong>
                  <span>Best {entry.bestFitness.toFixed(5)}</span>
                  <span>Mean {entry.meanFitness.toFixed(5)}</span>
                  <span>{entry.failedEvaluations} failed</span>
                </div>
              ))}
            </div>
          )}
        </section>
        <section className="panel best-candidate-panel">
          <div className="panel-kicker">BEST CANDIDATE</div>
          {best ? (
            <>
              <strong className="best-fitness">{best.fitness.toFixed(7)}</strong>
              <p>Generation {best.generation + 1} · Candidate {best.candidateIndex + 1}</p>
              <button className="primary-button full-button" onClick={() => { setNetwork(best.network); setWorkspace("network"); }}><BrainCircuit size={14} /> Load best network</button>
              <button className="secondary-button full-button" onClick={async () => {
                if (!best) return;
                try { await onSaveNetwork(best.network); setSaveMessage("Best candidate saved to Network Library."); }
                catch (saveError) { setSaveMessage(saveError instanceof Error ? saveError.message : String(saveError)); }
              }}><FolderOpen size={14} /> Save best network</button>
              {saveMessage && <p className="save-message">{saveMessage}</p>}
            </>
          ) : <p className="empty-logs">The best-performing network will appear here.</p>}
        </section>
      </div>

      <section className="panel candidate-history-panel">
        <div className="panel-heading compact"><div className="panel-kicker">CANDIDATE EVALUATIONS</div><span>{events.length} evaluated</span></div>
        {events.length === 0 ? <p className="empty-logs">Individual fitness results will appear here.</p> : (
          <div className="candidate-table">
            <div className="candidate-table-header"><span>Generation</span><span>Candidate</span><span>Fitness</span><span>Status</span></div>
            {events.slice(-30).reverse().map((event, index) => (
              <div className="candidate-table-row" key={`${event.generation}-${event.candidateIndex}-${index}`}>
                <span>{event.generation + 1}</span><span>{event.candidateIndex + 1}</span>
                <span>{event.error ? "—" : event.fitness.toFixed(6)}</span>
                <span className={event.error ? "candidate-failed" : "candidate-ok"}>{event.error ? "FAILED" : "OK"}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function PlaygroundOverview({
  network,
  setNetwork,
  source,
  setSource,
  onPlaygroundSaved,
  resourceRevision,
}: {
  network: MLPNetwork | null;
  setNetwork: (network: MLPNetwork | null) => void;
  source: string;
  setSource: (source: string) => void;
  onPlaygroundSaved: () => void;
  resourceRevision: number;
}) {
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "failed" | "cancelled">("idle");
  const [result, setResult] = useState<{ fitness: number; metrics?: Record<string, unknown> } | null>(null);
  const [logs, setLogs] = useState<Record<string, unknown>[]>([]);
  const [playbackDelay, setPlaybackDelay] = useState(80);
  const [environmentMaxSteps, setEnvironmentMaxSteps] = useState(500);
  const [environmentSeed, setEnvironmentSeed] = useState(42);
  const [playgroundName, setPlaygroundName] = useState("Custom Playground");
  const [playgroundId, setPlaygroundId] = useState("custom-playground");
  const [playgroundInputSize, setPlaygroundInputSize] = useState(network?.inputSize ?? 3);
  const [playgroundOutputSize, setPlaygroundOutputSize] = useState(network?.layers[network.layers.length - 1]?.units ?? 2);
  const [savedPlaygrounds, setSavedPlaygrounds] = useState<SavedPlaygroundRecord[]>([]);
  const [loadPlaygroundId, setLoadPlaygroundId] = useState("");
  const [savePlaygroundMessage, setSavePlaygroundMessage] = useState<string | null>(null);
  const [savePlaygroundError, setSavePlaygroundError] = useState<string | null>(null);
  const [playgroundParametersJson, setPlaygroundParametersJson] = useState("[]");
  useEffect(() => {
    let active = true;
    listSavedPlaygrounds().then((items) => {
      if (!active) return;
      setSavedPlaygrounds(items);
      setLoadPlaygroundId((current) => current || items[0]?.id || "");
    }).catch((e) => setSavePlaygroundError(e instanceof Error ? e.message : String(e)));
    return () => { active = false; };
  }, [resourceRevision]);
  useEffect(() => {
    if (source.includes('id: "snake"')) { setPlaygroundName("Snake"); setPlaygroundId("snake"); setPlaygroundInputSize(11); setPlaygroundOutputSize(3); }
  }, [source]);
  const saveCurrentPlayground = async () => {
    try {
      if (!source.trim()) throw new Error("Environment JavaScript cannot be empty.");
      if (!playgroundName.trim()) throw new Error("Enter a Playground name.");
      if (!/^[a-z0-9][a-z0-9._-]*$/i.test(playgroundId.trim())) throw new Error("Playground ID may contain letters, numbers, dots, underscores, and hyphens.");
      if (!Number.isInteger(playgroundInputSize) || playgroundInputSize < 1 || !Number.isInteger(playgroundOutputSize) || playgroundOutputSize < 1) throw new Error("Input/output dimensions must be positive integers.");
      const numberValue = (key: string, fallback: number) => Number(source.match(new RegExp(key + "\\s*:\\s*(\\d+)"))?.[1] ?? fallback);
      const parameters = JSON.parse(playgroundParametersJson) as PlaygroundDefinition["parameters"];
      if (!Array.isArray(parameters) || parameters.some((item) => !item || typeof item.key !== "string" || typeof item.label !== "string" || !["number", "boolean", "string"].includes(item.type) || !["number", "boolean", "string"].includes(typeof item.default))) throw new Error("Parameters must be a JSON array of typed parameter definitions.");
      const definition: PlaygroundDefinition = {
        id: playgroundId.trim(), name: playgroundName.trim(),
        version: numberValue("version", 1),
        inputSize: playgroundInputSize, outputSize: playgroundOutputSize, parameters,
      };
      await savePlayground(definition, source, playgroundName.trim());
      const items = await listSavedPlaygrounds();
      setSavedPlaygrounds(items);
      setLoadPlaygroundId(definition.id);
      onPlaygroundSaved();
      setSavePlaygroundError(null);
      setSavePlaygroundMessage("Saved. This Playground is now available in Training Studio.");
    } catch (e) {
      setSavePlaygroundMessage(null);
      setSavePlaygroundError(e instanceof Error ? e.message : String(e));
    }
  };
  const loadSavedPlayground = () => {
    const item = savedPlaygrounds.find((entry) => entry.id === loadPlaygroundId);
    if (!item) return;
    setSource(item.source);
    setPlaygroundName(item.name);
    setPlaygroundId(item.id);
    setPlaygroundInputSize(item.definition.inputSize);
    setPlaygroundOutputSize(item.definition.outputSize);
    setSavePlaygroundMessage("Loaded " + item.name + ".");
    setSavePlaygroundError(null);
    setResult(null); setLogs([]); setStatus("idle");
  };
  const [renderedView, setRenderedView] = useState<unknown>(source === snakeEnvironmentSource ? {
    type: "snake", width: 12, height: 12,
    snake: [{ x: 6, y: 6, head: true }, { x: 5, y: 6 }, { x: 4, y: 6 }],
    food: { x: 9, y: 3 }, score: 0, steps: 0, terminal: false,
  } : null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const runEnvironment = async () => {
    const activeNetwork = network ?? createMLP({
      name: "Playground sample network",
      inputSize: 2,
      layers: [{ units: 2, activation: "softmax" }],
    });
    if (!network) setNetwork(activeNetwork);

    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("running");
    setResult(null);
    setError(null);
    setLogs([]);
    setRenderedView(null);

    try {
      const outputSize = activeNetwork.layers[activeNetwork.layers.length - 1].units;
      const sessionResult = await runPlaygroundSession({
        source,
        network: activeNetwork,
        session: {
          sessionId: `playground-${Date.now()}`,
          generation: 0,
          candidateIndex: 0,
          seed: environmentSeed,
        },
        config: { inputSize: activeNetwork.inputSize, outputSize, stepDelayMs: playbackDelay, maxSteps: environmentMaxSteps },
        timeoutMs: 30_000,
        signal: controller.signal,
        onReport: (metrics) => setLogs((current) => [...current, metrics]),
        onRender: setRenderedView,
      });
      setResult(sessionResult);
      setStatus("completed");
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : String(runError));
      setStatus(controller.signal.aborted ? "cancelled" : "failed");
    } finally {
      abortRef.current = null;
    }
  };

  const cancelEnvironment = () => abortRef.current?.abort("Cancelled by user.");

  return (
    <div className="playground-editor-layout">
      <section className="panel environment-editor-panel">
        <div className="panel-heading">
          <div><div className="panel-kicker">CUSTOM ENVIRONMENT</div><h2>Environment module</h2></div>
          <div className="panel-tools">
            <span className="model-chip">JAVASCRIPT · ES MODULE</span>
            <button className="secondary-button" onClick={() => {
              setSource(snakeEnvironmentSource);
              setNetwork(createMLP({
                name: "Snake agent",
                inputSize: 11,
                layers: [{ units: 16, activation: "relu" }, { units: 3, activation: "softmax" }],
              }));
              setRenderedView({
                type: "snake", width: 12, height: 12,
                snake: [{ x: 6, y: 6, head: true }, { x: 5, y: 6 }, { x: 4, y: 6 }],
                food: { x: 9, y: 3 }, score: 0, steps: 0, terminal: false,
              });
              setResult(null);
              setLogs([]);
              setError(null);
              setStatus("idle");
            }}>Load Snake</button>
          </div>
        </div>
        <div className="playground-resource-controls">\n          <label>Playground name<input value={playgroundName} onChange={(e) => setPlaygroundName(e.target.value)} /></label>\n          <label>Stable ID<input value={playgroundId} onChange={(e) => setPlaygroundId(e.target.value)} /></label>\n          <label>Input size<input type="number" min="1" value={playgroundInputSize} onChange={(e) => setPlaygroundInputSize(e.target.valueAsNumber)} /></label>\n          <label>Output size<input type="number" min="1" value={playgroundOutputSize} onChange={(e) => setPlaygroundOutputSize(e.target.valueAsNumber)} /></label>\n          <label className="playground-parameters-field">Environment parameters (JSON)<textarea rows={3} value={playgroundParametersJson} onChange={(e) => setPlaygroundParametersJson(e.target.value)} spellCheck={false} /></label>
          <button className="primary-button" onClick={() => void saveCurrentPlayground()}><Save size={14} /> Save Playground</button>\n          <label>Saved Playground<select value={loadPlaygroundId} onChange={(e) => setLoadPlaygroundId(e.target.value)}><option value="">Select saved Playground…</option>{savedPlaygrounds.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.definition.inputSize}→{item.definition.outputSize}</option>)}</select></label>\n          <button className="secondary-button" onClick={loadSavedPlayground} disabled={!savedPlaygrounds.some((item) => item.id === loadPlaygroundId)}>Load selected</button>\n          {savePlaygroundMessage && <span className="save-message">{savePlaygroundMessage}</span>}\n          {savePlaygroundError && <span className="environment-error">{savePlaygroundError}</span>}\n        </div>\n        <div className="environment-editor-meta">
          <span>Exports <code>sessionStart(ctx)</code> and optional <code>sessionEnd(ctx, result)</code></span>
          <span>{source.split("\n").length} lines</span>
        </div>
        <div className="playground-howto">
          <strong>How to run this example</strong>
          <span>1. Click <b>Load Snake</b> to load the game and its matching 11-input / 3-output network.</span>
          <span>2. Click <b>Run environment</b> to watch one Snake session.</span>
          <span>3. Open <b>Training</b> and click <b>Start evolution</b> to train across a population.</span>
        </div>
        <textarea
          className="environment-code"
          spellCheck={false}
          value={source}
          onChange={(event) => setSource(event.target.value)}
          aria-label="Environment JavaScript source code"
        />
        <div className="environment-editor-footer">
          <div className="playback-speed-control playground-speed-control">
            <label htmlFor="playground-speed">Playback speed</label>
            <input id="playground-speed" type="range" min="0" max="500" step="10" disabled={status === "running"} value={playbackDelay} onChange={(event) => setPlaybackDelay(Number(event.target.value))} />
            <span>{playbackDelay === 0 ? "Fastest" : playbackDelay + " ms / step"}</span>
          </div>
          <label className="environment-run-parameter">Max steps
            <input type="number" min="1" max="10000" value={environmentMaxSteps} disabled={status === "running"} onChange={(event) => setEnvironmentMaxSteps(Math.max(1, event.target.valueAsNumber || 1))} />
          </label>
          <label className="environment-run-parameter">Environment seed
            <input type="number" step="1" value={environmentSeed} disabled={status === "running"} onChange={(event) => setEnvironmentSeed(event.target.valueAsNumber || 0)} />
          </label>
          <div className="selected-network-label">
            <BrainCircuit size={14} />
            <span>{network ? network.name : "A sample 2 → 2 MLP will be created on first run"}</span>
          </div>
          <div className="environment-actions">
            {status === "running" ? (
              <button className="secondary-button" onClick={cancelEnvironment}>Cancel session</button>
            ) : (
              <button className="primary-button" onClick={runEnvironment} disabled={!source.trim()}><Play size={14} /> Run environment</button>
            )}
          </div>
        </div>
      </section>

      <aside className="environment-results-column">
        <section className="panel environment-status-panel">
          <div className="panel-kicker">SESSION STATUS</div>
          <div className="session-status-line">
            <span className={`status-dot ${status === "completed" ? "good" : status === "failed" ? "warn" : status === "running" ? "good" : "neutral"}`} />
            <strong>{status === "idle" ? "Ready to run" : status === "running" ? "Running session" : status === "completed" ? "Session completed" : status === "cancelled" ? "Session cancelled" : "Session failed"}</strong>
          </div>
          {status === "running" && <p>Environment code is executing in a dedicated Worker. Network inference runs in the Worker using the JavaScript backend.</p>}
          {error && <div className="environment-error">{error}</div>}
          {result && (
            <div className="fitness-result">
              <span>FITNESS SCORE</span>
              <strong>{result.fitness.toFixed(6)}</strong>
              <small>{JSON.stringify(result.metrics ?? {})}</small>
            </div>
          )}
        </section>
        <section className="panel environment-view-panel">
          <div className="panel-heading compact"><div className="panel-kicker">ENVIRONMENT VIEW</div><span>{renderedView === null ? "waiting for ctx.render()" : "latest frame"}</span></div>
          <EnvironmentVisualization view={renderedView} />
        </section>
        <section className="panel environment-log-panel">
          <div className="panel-heading compact"><div className="panel-kicker">ENVIRONMENT REPORTS</div><span>{logs.length} events</span></div>
          <div className="environment-logs">
            {logs.length === 0 ? <p className="empty-logs">Progress metrics from <code>ctx.report()</code> will appear here.</p> : logs.map((entry, index) => (
              <div className="environment-log-entry" key={index}><span>{String(entry.step ?? index + 1).padStart(2, "0")}</span><code>{JSON.stringify(entry)}</code></div>
            ))}
          </div>
        </section>
        <section className="panel context-reference-panel">
          <div className="panel-kicker">AVAILABLE CONTEXT API</div>
          <code>ctx.network.predict(input)</code>
          <code>ctx.finish({"{ fitness, metrics }"})</code>
          <code>ctx.report(metrics) · ctx.render(view)</code>
          <code>ctx.random() · ctx.yield()</code>
          <code>ctx.signal · ctx.session · ctx.config</code>
          <a href="https://github.com/ckfun0skf/webnn-playground/blob/main/docs/PLAYGROUND_API.md" target="_blank" rel="noreferrer">Read environment API reference ↗</a>
        </section>
      </aside>
    </div>
  );
}

function NetworkLibrary({
  revision,
  onLoad,
}: {
  revision: number;
  onLoad: (network: MLPNetwork) => void;
}) {
  const [records, setRecords] = useState<SavedNetworkRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    listSavedNetworks()
      .then((items) => { if (active) setRecords(items); })
      .catch((loadError) => { if (active) setError(loadError instanceof Error ? loadError.message : String(loadError)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [revision]);

  const remove = async (id: string) => {
    try {
      await deleteSavedNetwork(id);
      setRecords((current) => current.filter((record) => record.id !== id));
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : String(deleteError));
    }
  };

  return (
    <section className="panel network-library-panel">
      <div className="panel-heading">
        <div><div className="panel-kicker">LOCAL MODEL STORAGE</div><h2>Saved networks</h2></div>
        <span className="model-chip">INDEXEDDB</span>
      </div>
      {loading ? <p className="empty-logs">Loading saved networks…</p> : records.length === 0 ? (
        <div className="library-empty"><FolderOpen size={25} /><strong>No saved networks yet</strong><p>Save a model from the Network workspace or keep the best candidate from a Neuroevolution run.</p></div>
      ) : (
        <div className="saved-network-list">
          {records.map((record) => (
            <div className="saved-network-row" key={record.id}>
              <div className="saved-network-icon"><BrainCircuit size={18} /></div>
              <div className="saved-network-copy">
                <strong>{record.name}</strong>
                <span>{record.network.inputSize} inputs · {record.network.layers.map((layer) => layer.units).join(" → ")} · Updated {new Date(record.updatedAt).toLocaleString()}</span>
              </div>
              <button className="secondary-button" onClick={() => onLoad(record.network)}>Load</button>
              <button className="small-icon-button" title="Delete saved network" aria-label="Delete saved network" onClick={() => void remove(record.id)}><span>×</span></button>
            </div>
          ))}
        </div>
      )}
      {error && <div className="environment-error">{error}</div>}
    </section>
  );
}

