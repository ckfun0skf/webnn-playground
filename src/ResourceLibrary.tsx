import { useEffect, useState } from "react";
import { Trash2, Save, BrainCircuit, FlaskConical, Workflow } from "lucide-react";
import { deleteSavedPlayground, deleteTrainingRun, listSavedPlaygrounds, listTrainingRuns, savePlayground, type SavedPlaygroundRecord, type SavedTrainingRunRecord } from "./storage";
import type { MLPNetwork } from "./core/model";
import type { PlaygroundDefinition } from "./playground/api";

type Props = {
  source: string;
  onLoadPlayground: (source: string) => void;
  onLoadNetwork: (network: MLPNetwork) => void;
  revision: number;
  onChanged: () => void;
};

function inferDefinition(source: string, name: string, id: string, inputSize: number, outputSize: number): PlaygroundDefinition {
  const readNumber = (key: string, fallback: number) => {
    const match = source.match(new RegExp(key + "\\s*:\\s*(\\d+)"));
    return match ? Number(match[1]) : fallback;
  };
  const readString = (key: string, fallback: string) => {
    const match = source.match(new RegExp(key + "\\s*:\\s*[\"']([^\"']+)[\"']"));
    return match ? match[1] : fallback;
  };
  return {
    id: readString("id", id),
    name: readString("name", name),
    version: readNumber("version", 1),
    inputSize: readNumber("inputSize", inputSize),
    outputSize: readNumber("outputSize", outputSize),
    parameters: [],
  };
}

export default function ResourceLibrary({ source, onLoadPlayground, onLoadNetwork, revision, onChanged }: Props) {
  const [playgrounds, setPlaygrounds] = useState<SavedPlaygroundRecord[]>([]);
  const [runs, setRuns] = useState<SavedTrainingRunRecord[]>([]);
  const [name, setName] = useState("Custom Playground");
  const [id, setId] = useState("custom-playground");
  const [inputSize, setInputSize] = useState(3);
  const [outputSize, setOutputSize] = useState(2);
  const [parametersJson, setParametersJson] = useState("[]");
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([listSavedPlaygrounds(), listTrainingRuns()]).then(([p, r]) => {
      if (active) { setPlaygrounds(p); setRuns(r); }
    }).catch((e) => { if (active) setError(e instanceof Error ? e.message : String(e)); });
    return () => { active = false; };
  }, [revision]);

  useEffect(() => {
    const def = inferDefinition(source, name, id, inputSize, outputSize);
    setName(def.name); setId(def.id); setInputSize(def.inputSize); setOutputSize(def.outputSize);
  }, [source]);

  const saveCurrent = async () => {
    try {
      if (!source.trim()) throw new Error("Playground source cannot be empty.");
      if (!/^[a-z0-9][a-z0-9._-]*$/i.test(id.trim())) throw new Error("ID may contain letters, numbers, dots, underscores, and hyphens.");
      if (!Number.isInteger(inputSize) || inputSize < 1 || !Number.isInteger(outputSize) || outputSize < 1) throw new Error("Input and output sizes must be positive integers.");
      const definition = inferDefinition(source, name.trim(), id.trim(), inputSize, outputSize);
      const parameters = JSON.parse(parametersJson);
      if (!Array.isArray(parameters) || parameters.some((item) => !item || typeof item.key !== "string" || typeof item.label !== "string" || !["number", "boolean", "string"].includes(item.type) || !["number", "boolean", "string"].includes(typeof item.default))) throw new Error("Parameters must be a JSON array of typed parameter definitions.");
      definition.parameters = parameters;
      definition.id = id.trim(); definition.name = name.trim(); definition.inputSize = inputSize; definition.outputSize = outputSize;
      await savePlayground(definition, source, name.trim());
      onChanged(); setMessage("Playground saved to the local library."); setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  const removePlayground = async (record: SavedPlaygroundRecord) => {
    if (!window.confirm("Delete Playground " + record.name + "?")) return;
    try { await deleteSavedPlayground(record.id); setPlaygrounds((items) => items.filter((x) => x.id !== record.id)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const removeRun = async (run: SavedTrainingRunRecord) => {
    if (!window.confirm("Delete Training Run " + run.name + "?")) return;
    try { await deleteTrainingRun(run.id); setRuns((items) => items.filter((x) => x.id !== run.id)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  return <div className="resource-library">
    <section className="panel">
      <div className="panel-heading"><div><div className="panel-kicker">PLAYGROUND LIBRARY</div><h2>Save the current Playground</h2></div><FlaskConical size={18} /></div>
      <p className="resource-help">A Playground stores its JavaScript source and the dimensions required by compatible Neural Networks.</p>
      <div className="resource-form-grid">
        <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
        <label>Stable ID<input value={id} onChange={(e) => setId(e.target.value)} /></label>
        <label>Input dimensions<input type="number" min="1" value={inputSize} onChange={(e) => setInputSize(e.target.valueAsNumber)} /></label>
        <label>Output dimensions<input type="number" min="1" value={outputSize} onChange={(e) => setOutputSize(e.target.valueAsNumber)} /></label>
      </div>
      <label className="resource-parameters-label">Typed parameters (JSON array)<textarea rows={4} value={parametersJson} onChange={(e) => setParametersJson(e.target.value)} spellCheck={false} /></label>
      <button className="primary-button" onClick={() => void saveCurrent()}><Save size={14} /> Save Playground</button>
      {message && <p className="save-message">{message}</p>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><div className="panel-kicker">REUSABLE ENVIRONMENTS</div><h2>Saved Playgrounds</h2></div><span className="model-chip">{playgrounds.length} SAVED</span></div>
      {playgrounds.length === 0 ? <p className="empty-logs">No saved Playgrounds yet. Save the current environment above.</p> : <div className="saved-network-list">{playgrounds.map((p) => <div className="saved-network-row" key={p.id}>
        <div className="saved-network-icon"><FlaskConical size={18} /></div><div className="saved-network-copy"><strong>{p.name}</strong><span>{p.definition.inputSize} inputs · {p.definition.outputSize} outputs · v{p.version} · {p.id}</span></div>
        <button className="secondary-button" onClick={() => onLoadPlayground(p.source)}>Load</button><button className="small-icon-button" title="Delete Playground" onClick={() => void removePlayground(p)}><Trash2 size={14} /></button>
      </div>)}</div>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><div className="panel-kicker">EXPERIMENT HISTORY</div><h2>Training Runs</h2></div><Workflow size={18} /></div>
      {runs.length === 0 ? <p className="empty-logs">Completed training runs will appear here.</p> : <div className="saved-network-list">{runs.map((run) => <div className="saved-network-row" key={run.id}>
        <div className="saved-network-icon"><BrainCircuit size={18} /></div><div className="saved-network-copy"><strong>{run.name}</strong><span>{run.playgroundName} · {run.status} · Best fitness {run.bestFitness?.toFixed(4) ?? "—"} · {new Date(run.createdAt).toLocaleString()}</span></div>
        {run.bestNetwork && <button className="secondary-button" onClick={() => onLoadNetwork(run.bestNetwork!)}>Load best network</button>}
        <button className="small-icon-button" title="Delete Training Run" onClick={() => void removeRun(run)}><Trash2 size={14} /></button>
      </div>)}</div>}
    </section>
    {error && <div className="environment-error">{error}</div>}
  </div>;
}
