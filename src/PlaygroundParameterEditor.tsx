import type { PlaygroundParameterDefinition } from "./playground/api";

type Props = {
  parameters: PlaygroundParameterDefinition[];
  onChange: (parameters: PlaygroundParameterDefinition[]) => void;
};

export default function PlaygroundParameterEditor({ parameters, onChange }: Props) {
  const update = (index: number, patch: Partial<PlaygroundParameterDefinition>) => {
    onChange(parameters.map((parameter, itemIndex) => itemIndex === index ? { ...parameter, ...patch } : parameter));
  };
  const add = () => onChange([...parameters, {
    key: "parameter" + (parameters.length + 1),
    label: "New parameter",
    type: "number",
    default: 0,
  }]);
  return <div className="playground-parameter-editor">
    {parameters.map((parameter, index) => <div className="playground-parameter-row" key={index}>
      <div className="playground-parameter-fields">
        <label>Key<input value={parameter.key} onChange={(event) => update(index, { key: event.target.value })} /></label>
        <label>Label<input value={parameter.label} onChange={(event) => update(index, { label: event.target.value })} /></label>
        <label>Type<select value={parameter.type} onChange={(event) => {
          const type = event.target.value as PlaygroundParameterDefinition["type"];
          update(index, { type, default: type === "boolean" ? false : type === "string" ? "" : 0 });
        }}>
          <option value="number">Number</option><option value="boolean">Boolean</option><option value="string">String</option><option value="randomSeed">Random seed</option>
        </select></label>
        <label>Default{parameter.type === "boolean" ? <input type="checkbox" checked={Boolean(parameter.default)} onChange={(event) => update(index, { default: event.target.checked })} /> : <input type={parameter.type === "number" || parameter.type === "randomSeed" ? "number" : "text"} value={String(parameter.default)} onChange={(event) => update(index, { default: parameter.type === "number" || parameter.type === "randomSeed" ? event.target.valueAsNumber : event.target.value })} />}</label>
        {(parameter.type === "number") && <>
          <label>Min<input type="number" value={parameter.min ?? ""} onChange={(event) => update(index, { min: event.target.value === "" ? undefined : event.target.valueAsNumber })} /></label>
          <label>Max<input type="number" value={parameter.max ?? ""} onChange={(event) => update(index, { max: event.target.value === "" ? undefined : event.target.valueAsNumber })} /></label>
          <label>Step<input type="number" min="0.000001" value={parameter.step ?? ""} onChange={(event) => update(index, { step: event.target.value === "" ? undefined : event.target.valueAsNumber })} /></label>
        </>}
        <label className="parameter-description">Description<input value={parameter.description ?? ""} onChange={(event) => update(index, { description: event.target.value })} /></label>
      </div>
      <button className="text-button" onClick={() => onChange(parameters.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
    </div>)}
    <button className="secondary-button" onClick={add}><span>+</span> Add parameter</button>
  </div>;
}
