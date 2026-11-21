import { useMemo, useRef, useState } from "react";
import { createMLP, updateLayerParameters, type MLPNetwork } from "./core/model";
import { compileMLP } from "./core/engine";
import { runPlaygroundSession } from "./playground/runtime";
import {
  Activity,
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

type Workspace = "network" | "training" | "playground" | "library";

const navigation: { id: Workspace; label: string; icon: typeof BrainCircuit }[] = [
  { id: "network", label: "Network", icon: BrainCircuit },
  { id: "training", label: "Training", icon: Activity },
  { id: "playground", label: "Playground", icon: FlaskConical },
  { id: "library", label: "Network Library", icon: FolderOpen },
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
  const [inputText, setInputText] = useState("0, 0, 0");
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const capabilities = useMemo(detectCapabilities, []);

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

        {workspace === "network" && <NetworkOverview network={network} setNetwork={setNetwork} inputText={inputText} setInputText={setInputText} prediction={prediction} setPrediction={setPrediction} inferenceError={inferenceError} setInferenceError={setInferenceError} />}
        {workspace === "training" && <ComingSoon icon={Activity} title="Training Studio" description="Train networks with supervised learning and neuroevolution. Training runs, candidate evaluation, and fitness history will live here." />}
        {workspace === "playground" && <PlaygroundOverview network={network} setNetwork={setNetwork} />}
        {workspace === "library" && <ComingSoon icon={FolderOpen} title="Network Library" description="Saved networks and selected evolution candidates will be available here, ready to reload into inference or a Playground." />}
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
  }
}

function NetworkOverview({
  network,
  setNetwork,
  inputText,
  setInputText,
  prediction,
  setPrediction,
  inferenceError,
  setInferenceError,
}: {
  network: MLPNetwork | null;
  setNetwork: (network: MLPNetwork | null) => void;
  inputText: string;
  setInputText: (value: string) => void;
  prediction: number[] | null;
  setPrediction: (value: number[] | null) => void;
  inferenceError: string | null;
  setInferenceError: (value: string | null) => void;
}) {
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
            <div className="model-detail-heading"><strong>Layer parameters</strong><span>{network.layers.length} dense layers</span></div>
            {network.layers.map((layer, index) => (
              <div className="layer-editor" key={layer.id}>
                <div className="layer-detail">
                  <div><strong>Layer {index + 1}</strong><span>{layer.units} units</span></div>
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

function PlaygroundOverview({
  network,
  setNetwork,
}: {
  network: MLPNetwork | null;
  setNetwork: (network: MLPNetwork | null) => void;
}) {
  const [source, setSource] = useState(DEFAULT_ENVIRONMENT_CODE);
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "failed">("idle");
  const [result, setResult] = useState<{ fitness: number; metrics?: Record<string, unknown> } | null>(null);
  const [logs, setLogs] = useState<Record<string, unknown>[]>([]);
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

    try {
      const outputSize = activeNetwork.layers[activeNetwork.layers.length - 1].units;
      const sessionResult = await runPlaygroundSession({
        source,
        network: activeNetwork,
        session: {
          sessionId: `playground-${Date.now()}`,
          generation: 0,
          candidateIndex: 0,
          seed: 42,
        },
        config: { inputSize: activeNetwork.inputSize, outputSize },
        timeoutMs: 30_000,
        signal: controller.signal,
        onReport: (metrics) => setLogs((current) => [...current, metrics]),
      });
      setResult(sessionResult);
      setStatus("completed");
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : String(runError));
      setStatus("failed");
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
          <span className="model-chip">JAVASCRIPT · ES MODULE</span>
        </div>
        <div className="environment-editor-meta">
          <span>Exports <code>sessionStart(ctx)</code> and optional <code>sessionEnd(ctx, result)</code></span>
          <span>{source.split("\n").length} lines</span>
        </div>
        <textarea
          className="environment-code"
          spellCheck={false}
          value={source}
          onChange={(event) => setSource(event.target.value)}
          aria-label="Environment JavaScript source code"
        />
        <div className="environment-editor-footer">
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
            <strong>{status === "idle" ? "Ready to run" : status === "running" ? "Running session" : status === "completed" ? "Session completed" : "Session failed"}</strong>
          </div>
          {status === "running" && <p>Environment code is executing in a dedicated Worker. Network inference is handled by the JavaScript backend.</p>}
          {error && <div className="environment-error">{error}</div>}
          {result && (
            <div className="fitness-result">
              <span>FITNESS SCORE</span>
              <strong>{result.fitness.toFixed(6)}</strong>
              <small>{JSON.stringify(result.metrics ?? {})}</small>
            </div>
          )}
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
          <code>ctx.report(metrics)</code>
          <code>ctx.random() · ctx.yield()</code>
          <code>ctx.signal · ctx.session · ctx.config</code>
          <a href="https://github.com/ckfun0skf/webnn-playground/blob/main/docs/PLAYGROUND_API.md" target="_blank" rel="noreferrer">Read environment API reference ↗</a>
        </section>
      </aside>
    </div>
  );
}

function ComingSoon({ icon: Icon, title, description }: { icon: typeof BrainCircuit; title: string; description: string }) {
  return (
    <section className="panel coming-soon">
      <div className="coming-icon"><Icon size={22} /></div>
      <span className="coming-tag">PLANNED</span>
      <h2>{title}</h2>
      <p>{description}</p>
    </section>
  );
}
