import { useState } from "react";
import { createMLP, type MLPNetwork } from "./core/model";

type LayerDraft = {
  units: number;
  activation: MLPNetwork["layers"][number]["activation"];
  useBias: boolean;
};

type Props = {
  defaultName: string;
  existingNames: string[];
  fixedInputSize?: number;
  fixedOutputSize?: number;
  onClose: () => void;
  onCreate: (network: MLPNetwork) => Promise<void>;
};

function uniqueName(base: string, existing: string[]): string {
  const names = new Set(existing.map((name) => name.toLowerCase()));
  if (!names.has(base.toLowerCase())) return base;
  let suffix = 1;
  while (names.has((base + " (" + suffix + ")").toLowerCase())) suffix += 1;
  return base + " (" + suffix + ")";
}

export default function NetworkBuilderDialog({
  defaultName,
  existingNames,
  fixedInputSize,
  fixedOutputSize,
  onClose,
  onCreate,
}: Props) {
  const [name, setName] = useState(defaultName);
  const [inputSize, setInputSize] = useState(fixedInputSize ?? 3);
  const [outputSize, setOutputSize] = useState(fixedOutputSize ?? 2);
  const [layers, setLayers] = useState<LayerDraft[]>([
    { units: 16, activation: "relu", useBias: true },
    { units: fixedOutputSize ?? 2, activation: "softmax", useBias: true },
  ]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateLayer = (index: number, update: Partial<LayerDraft>) => {
    setLayers((current) => current.map((layer, layerIndex) => layerIndex === index ? { ...layer, ...update } : layer));
  };
  const addHiddenLayer = () => {
    setLayers((current) => [...current.slice(0, -1), { units: 16, activation: "relu", useBias: true }, current[current.length - 1]]);
  };
  const removeHiddenLayer = (index: number) => {
    if (index >= layers.length - 1) return;
    setLayers((current) => current.filter((_, layerIndex) => layerIndex !== index));
  };
  const create = async () => {
    try {
      if (!name.trim()) throw new Error("Enter a Network name.");
      if (!Number.isInteger(inputSize) || inputSize < 1 || !Number.isInteger(outputSize) || outputSize < 1) throw new Error("Input and output node counts must be positive integers.");
      const definitions = layers.map((layer, index) => ({
        units: index === layers.length - 1 ? outputSize : layer.units,
        activation: layer.activation,
        useBias: layer.useBias,
      }));
      const network = createMLP({
        name: uniqueName(name.trim(), existingNames),
        inputSize,
        layers: definitions,
      });
      setSaving(true);
      await onCreate(network);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : String(createError));
      setSaving(false);
    }
  };

  return <div className="modal-backdrop" role="presentation">
    <section className="panel network-builder-dialog" role="dialog" aria-modal="true" aria-labelledby="network-builder-title">
      <div className="panel-heading">
        <div><div className="panel-kicker">NETWORK ARCHITECTURE</div><h2 id="network-builder-title">Create Network</h2></div>
        <button className="small-icon-button" onClick={onClose} aria-label="Close dialog">×</button>
      </div>
      <label>Network name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <div className="network-builder-dimensions">
        <label>Input layer · nodes<input type="number" min="1" max="512" value={fixedInputSize ?? inputSize} disabled={fixedInputSize !== undefined} onChange={(event) => setInputSize(event.target.valueAsNumber)} /></label>
        <label>Output layer · nodes<input type="number" min="1" max="512" value={fixedOutputSize ?? outputSize} disabled={fixedOutputSize !== undefined} onChange={(event) => setOutputSize(event.target.valueAsNumber)} /></label>
      </div>
      <div className="network-builder-layer-list">
        {layers.map((layer, index) => {
          const output = index === layers.length - 1;
          return <div className="network-builder-layer" key={index}>
            <div className="network-builder-layer-title"><strong>{output ? "Output layer" : "Hidden layer " + (index + 1)}</strong>{!output && <button className="text-button" onClick={() => removeHiddenLayer(index)}>Remove</button>}</div>
            <label>Nodes<input type="number" min="1" max="512" value={output ? outputSize : layer.units} disabled={output && fixedOutputSize !== undefined} onChange={(event) => updateLayer(index, { units: event.target.valueAsNumber })} /></label>
            <label>Activation<select value={layer.activation} onChange={(event) => updateLayer(index, { activation: event.target.value as LayerDraft["activation"] })}>{["linear", "relu", "tanh", "sigmoid", "softmax"].map((activation) => <option key={activation}>{activation}</option>)}</select></label>
            <label className="network-builder-bias"><input type="checkbox" checked={layer.useBias} onChange={(event) => updateLayer(index, { useBias: event.target.checked })} /> Use bias</label>
          </div>;
        })}
      </div>
      <button className="secondary-button" onClick={addHiddenLayer}>+ Add hidden layer</button>
      {error && <p className="environment-error">{error}</p>}
      <div className="network-builder-actions"><button className="secondary-button" onClick={onClose} disabled={saving}>Cancel</button><button className="primary-button" onClick={() => void create()} disabled={saving}>{saving ? "Saving…" : "Create and save Network"}</button></div>
    </section>
  </div>;
}
