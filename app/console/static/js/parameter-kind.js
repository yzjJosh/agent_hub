export function parameterKind(param, value) {
  if (!param) {
    if (Array.isArray(value)) return value.every((item) => typeof item === "string") ? "string-list" : "raw";
    if (typeof value === "string") return "text";
    return ["number", "boolean"].includes(typeof value) ? "scalar" : "raw";
  }
  const { type, item_type: itemType, kind } = param;
  if (type === "array") {
    if (itemType !== "string") return "raw";
    if (kind === "resource_list") return "resource-list";
    return !kind || kind === "text" ? "string-list" : "raw";
  }
  if (type === "string") {
    if (kind === "resource") return "resource";
    if (!kind || ["text", "longtext", "url", "date"].includes(kind)) return "text";
    return "raw";
  }
  return ["integer", "number", "boolean"].includes(type) && !kind ? "scalar" : "raw";
}
