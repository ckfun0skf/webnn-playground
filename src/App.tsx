import { useMemo, useState } from "react";
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

        {workspace === "network" && <NetworkOverview />}
        {workspace === "training" && <ComingSoon icon={Activity} title="Training Studio" description="Train networks with supervised learning and neuroevolution. Training runs, candidate evaluation, and fitness history will live here." />}
        {workspace === "playground" && <PlaygroundOverview />}
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

function NetworkOverview() {
  return (
    <div className="content-grid">
      <section className="panel network-panel">
        <div className="panel-heading">
          <div><div className="panel-kicker">MODEL CANVAS</div><h2>Network architecture</h2></div>
          <div className="panel-tools"><span className="model-chip">MLP</span><button className="small-icon-button" aria-label="More options"><Settings2 size={15} /></button></div>
        </div>
        <div className="empty-canvas">
          <div className="canvas-grid" />
          <div className="network-placeholder">
            <div className="layer-column">
              <span className="layer-label">INPUT</span>
              {[0, 1, 2].map((n) => <span className="neuron input-neuron" key={n}>{["x₁", "x₂", "x₃"][n]}</span>)}
            </div>
            <div className="connection-lines"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
            <div className="layer-column">
              <span className="layer-label">HIDDEN</span>
              {[0, 1, 2, 3].map((n) => <span className="neuron hidden-neuron" key={n}>h{n + 1}</span>)}
            </div>
            <div className="connection-lines"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
            <div className="layer-column">
              <span className="layer-label">OUTPUT</span>
              {[0, 1].map((n) => <span className="neuron output-neuron" key={n}>y{n + 1}</span>)}
            </div>
          </div>
          <div className="canvas-empty-caption"><Sparkles size={13} /> Your network canvas will become editable here</div>
        </div>
        <div className="canvas-toolbar">
          <span><span className="legend-dot input" /> Input <span className="legend-dot hidden" /> Hidden <span className="legend-dot output" /> Output</span>
          <span>Canvas · 100%</span>
        </div>
      </section>

      <aside className="right-column">
        <section className="panel quick-panel">
          <div className="panel-kicker">GET STARTED</div>
          <h2>Create your first network</h2>
          <p>Start with a simple multilayer perceptron. Add layers, configure neurons, and inspect weights and biases.</p>
          <button className="primary-button full-button"><Plus size={15} /> Create MLP network</button>
          <button className="secondary-button full-button"><Code2 size={15} /> Open network code</button>
        </section>
        <section className="panel backend-panel">
          <div className="panel-heading compact"><div className="panel-kicker">EXECUTION BACKENDS</div><Gauge size={16} /></div>
          <div className="backend-row"><span className="backend-icon js">JS</span><div><strong>JavaScript</strong><small>Baseline · always available</small></div><span className="backend-state available">READY</span></div>
          <div className="backend-row"><span className="backend-icon webnn"><Cpu size={16} /></span><div><strong>WebNN</strong><small>Optional · browser dependent</small></div><span className={`backend-state ${"ml" in navigator ? "available" : "unavailable"}`}>{ "ml" in navigator ? "DETECTED" : "UNAVAILABLE"}</span></div>
        </section>
      </aside>
    </div>
  );
}

function PlaygroundOverview() {
  return (
    <div className="playground-layout">
      <section className="panel playground-intro">
        <div className="intro-icon"><FlaskConical size={22} /></div>
        <div className="panel-kicker">GENERIC SIMULATION FRAMEWORK</div>
        <h2>Your environment. Your rules. Your neural network.</h2>
        <p>Write JavaScript to define a simulation. The Playground runtime manages sessions, connects the current candidate network, and collects fitness for neuroevolution.</p>
        <div className="contract-preview">
          <div className="code-line"><span className="code-purple">export async function</span> <span className="code-yellow">sessionStart</span>(ctx) {"{"}</div>
          <div className="code-line indent"><span className="code-muted">// Ask the current network for an action</span></div>
          <div className="code-line indent"><span className="code-purple">const</span> action = <span className="code-purple">await</span> ctx.network.predict(observation);</div>
          <div className="code-line indent">...</div>
          <div className="code-line indent">ctx.finish({"{"} fitness, metrics {"}"});</div>
          <div className="code-line">{"}"}</div>
        </div>
        <button className="primary-button"><Code2 size={15} /> Create environment</button>
      </section>
      <section className="panel templates-panel">
        <div className="panel-heading"><div><div className="panel-kicker">START WITH AN EXAMPLE</div><h2>Environment templates</h2></div><span className="coming-tag">IN DEVELOPMENT</span></div>
        <div className="template-card">
          <div className="template-art snake-art"><span className="snake-cell" /><span className="snake-cell" /><span className="snake-cell" /><span className="food-cell" /></div>
          <div className="template-copy"><strong>Snake</strong><p>Train an agent to navigate, collect food, and avoid collisions.</p><span>NEUROEVOLUTION · GRID SIMULATION</span></div>
          <button className="small-icon-button" aria-label="Open Snake template"><Play size={15} /></button>
        </div>
        <div className="template-card">
          <div className="template-art abstract-art"><span /><span /><span /><span /></div>
          <div className="template-copy"><strong>Minimal environment</strong><p>A tiny reference simulation for learning the environment API.</p><span>STARTER · CUSTOM CODE</span></div>
          <button className="small-icon-button" aria-label="Open starter template"><Play size={15} /></button>
        </div>
      </section>
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
