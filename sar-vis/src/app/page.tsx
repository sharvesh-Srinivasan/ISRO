"use client";

import { useState, useRef } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, Cell } from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────
type Bounds = { left: number; bottom: number; right: number; top: number };
type FileMetadata = {
  filename: string; size_mb: number; dtype: string; band_count: number;
  resolution: string; total_pixels: number; crs: string;
  bounds: Bounds; transform: number[];
};
type ArrayStats = { min: number; max: number; mean: number; std: number; variance: number; median: number; p5: number; p95: number };
type DbStats   = { mean: number; std: number; min: number; max: number; range: number };
type BandConfig = {
  band: string; lower_db: number; upper_db: number;
  water_threshold_db: number; land_threshold_db: number; crop_threshold_db: number;
  freq_ghz: string; wavelength_cm: string; penetration: string; application: string;
};
type Classification = {
  water_pct: number; land_pct: number; crop_pct: number; mountain_pct: number;
  water_px: number; land_px: number; crop_px: number; mountain_px: number;
};
type QualityMetrics = { snr: number; entropy_bits: number; process_ms: number; dynamic_range_db: number };
type ModelMetric = { Model: string; Accuracy: number; Mean_IoU: number; F1_Score: number; Latency_ms: number };
type ProcessData = {
  images: { raw: string; rgb: string; stretched: string; mask: string; unet: string; cnn: string; vision: string; rf: string };
  file_metadata: FileMetadata;
  raw_array_stats: ArrayStats;
  db_stats: DbStats;
  band_config: BandConfig;
  classification: Classification;
  quality_metrics: QualityMetrics;
  histogram: { bin: number; count: number }[];
  raw_histogram: { bin: number; count: number }[];
  synth: { method: string; desc: string; raw_label: string };
  model_metrics: ModelMetric[];
};

// ── Class legend ──────────────────────────────────────────────────────────────
const CLASSES = [
  { key: "water",    label: "Water Bodies",      color: "#2980b9", pctKey: "water_pct",    pxKey: "water_px"    },
  { key: "land",     label: "Vegetation / Land",  color: "#27ae60", pctKey: "land_pct",     pxKey: "land_px"     },
  { key: "crop",     label: "Agricultural Crop",  color: "#e67e22", pctKey: "crop_pct",     pxKey: "crop_px"     },
  { key: "mountain", label: "Mountainous Terrain",color: "#c0392b", pctKey: "mountain_pct", pxKey: "mountain_px" },
] as const;

export default function Home() {
  const [band, setBand]               = useState<"L" | "C" | "S">("C");
  const [hhPath, setHhPath]           = useState<string>("D:\\ISRO\\Proj\\C_Band\\E04_HH_tiles");
  const [hvPath, setHvPath]           = useState<string>("D:\\ISRO\\Proj\\C_Band\\E04_HV_tiles");
  const [state, setState]             = useState<"IDLE" | "PROCESSING" | "COMPLETE">("IDLE");
  const [data, setData]               = useState<ProcessData | null>(null);
  const [error, setError]             = useState("");
  const [tab, setTab]                 = useState<"RAW" | "RGB" | "STRETCHED" | "MASK" | "UNET" | "CNN" | "VISION" | "RF">("RAW");

  const run = async () => {
    if (!hhPath || !hvPath) { setError("Both HH and HV paths must be provided."); return; }
    setError(""); setState("PROCESSING");
    const fd = new FormData();
    fd.append("hh_path", hhPath);
    fd.append("hv_path", hvPath);
    fd.append("band", band);
    fd.append("run_models", "true");
    try {
      const res  = await fetch("http://127.0.0.1:5000/api/process", { 
        method: "POST", 
        body: fd,
        headers: {
          "Bypass-Tunnel-Reminder": "true"
        }
      });
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json); setState("COMPLETE"); setTab("RAW");
      setTimeout(() => setTab("RGB"),       1500);
      setTimeout(() => setTab("STRETCHED"), 3000);
      setTimeout(() => setTab("MASK"),      5000);
    } catch (e: any) { setError(e.message); setState("IDLE"); }
  };

  const reset = () => { setData(null); setState("IDLE"); setError(""); };

  return (
    <div className="root">
      {/* ── NAV ─────────────────────────────────────────── */}
      <nav className="nav">
        <div className="nav-left">
          <span className="nav-logo">◈</span>
          <span className="nav-title">SAR Analytics Platform</span>
          <span className="nav-sep">|</span>
          <span className="nav-sub">v5.0 · NISAR · ISRO</span>
        </div>
        <div className="nav-right">
          <div className="status-dot"></div>
          <span className="status-txt">Engine Ready</span>
          <div className="band-toggle">
            {(["L","C","S"] as const).map(b => (
              <button key={b} onClick={() => { 
                setBand(b); 
                if (b === "C") {
                  setHhPath("D:\\ISRO\\Proj\\C_Band\\E04_HH_tiles");
                  setHvPath("D:\\ISRO\\Proj\\C_Band\\E04_HV_tiles");
                } else if (b === "L") {
                  setHhPath("D:\\ISRO\\Proj\\HH.tif");
                  setHvPath("D:\\ISRO\\Proj\\HV.tif");
                } else {
                  setHhPath("");
                  setHvPath("");
                }
                reset(); 
              }}
                className={b === band ? "band-btn active" : "band-btn"}>
                {b}-Band
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* ── BODY ────────────────────────────────────────── */}
      <div className="body">

        {/* ── LEFT SIDEBAR ────────────────────────────── */}
        <aside className="sidebar">

          {/* Upload */}
          <section className="card">
            <div className="card-header">DIRECT DISK INGESTION</div>
            <div style={{ padding: "15px" }}>
              <label style={{ display: "block", marginBottom: "10px", fontSize: "0.9rem", color: "#ccc" }}>
                <strong>Absolute HH Path:</strong>
                <input 
                  type="text" 
                  value={hhPath} 
                  onChange={(e) => setHhPath(e.target.value)} 
                  placeholder="e.g. D:\ISRO\Proj\L_Band\HH_tiles"
                  style={{ width: "100%", padding: "10px", marginTop: "5px", background: "#1a1a1a", color: "white", border: "1px solid #333", borderRadius: "6px", fontFamily: "monospace" }} 
                />
              </label>
              <label style={{ display: "block", marginBottom: "15px", fontSize: "0.9rem", color: "#ccc" }}>
                <strong>Absolute HV Path:</strong>
                <input 
                  type="text" 
                  value={hvPath} 
                  onChange={(e) => setHvPath(e.target.value)} 
                  placeholder="e.g. D:\ISRO\Proj\L_Band\HV_tiles"
                  style={{ width: "100%", padding: "10px", marginTop: "5px", background: "#1a1a1a", color: "white", border: "1px solid #333", borderRadius: "6px", fontFamily: "monospace" }} 
                />
              </label>
            </div>
            <button className={state === "PROCESSING" ? "run-btn running" : "run-btn"}
              onClick={run} disabled={state === "PROCESSING" || !hhPath || !hvPath}>
              {state === "PROCESSING" ? <><span className="spinner"></span> Processing…</> : "▶  Run Pipeline"}
            </button>
            {data && <button className="reset-btn" onClick={reset}>↺  Reset</button>}
            {error && <div className="error-box">{error}</div>}
          </section>

          {/* Band parameters */}
          {data && (
            <section className="card">
              <div className="card-header">BAND CONFIGURATION · {band}-BAND</div>
              <div className="kv-grid">
                <KV k="Frequency"      v={data.band_config.freq_ghz} />
                <KV k="Wavelength"     v={data.band_config.wavelength_cm + " cm"} />
                <KV k="Stretch Lower"  v={data.band_config.lower_db + " dB"} />
                <KV k="Stretch Upper"  v={data.band_config.upper_db + " dB"} />
                <KV k="Water Thresh."  v={"≤ " + data.band_config.water_threshold_db + " dB"} />
                <KV k="Land Thresh."   v={"≤ " + data.band_config.land_threshold_db + " dB"} />
                <KV k="Crop Thresh."   v={"≤ " + data.band_config.crop_threshold_db + " dB"} />
              </div>
              <div className="info-block">
                <p className="info-label">PENETRATION</p>
                <p className="info-val">{data.band_config.penetration}</p>
              </div>
              <div className="info-block">
                <p className="info-label">APPLICATIONS</p>
                <p className="info-val">{data.band_config.application}</p>
              </div>
            </section>
          )}

          {/* File metadata */}
          {data && (
            <section className="card">
              <div className="card-header">FILE METADATA</div>
              <div className="kv-grid">
                <KV k="Filename"   v={data.file_metadata.filename} mono />
                <KV k="Size"       v={data.file_metadata.size_mb + " MB"} />
                <KV k="Data Type"  v={data.file_metadata.dtype} mono />
                <KV k="Bands"      v={String(data.file_metadata.band_count)} />
                <KV k="Resolution" v={data.file_metadata.resolution + " px"} />
                <KV k="Pixels"     v={data.file_metadata.total_pixels.toLocaleString()} />
                <KV k="CRS"        v={data.file_metadata.crs} mono />
              </div>
              <div className="info-block">
                <p className="info-label">BOUNDING BOX (WGS84)</p>
                <div className="kv-grid">
                  <KV k="Left"   v={String(data.file_metadata.bounds.left)} mono />
                  <KV k="Right"  v={String(data.file_metadata.bounds.right)} mono />
                  <KV k="Top"    v={String(data.file_metadata.bounds.top)} mono />
                  <KV k="Bottom" v={String(data.file_metadata.bounds.bottom)} mono />
                </div>
              </div>
              <div className="info-block">
                <p className="info-label">AFFINE TRANSFORM MATRIX</p>
                <code className="transform-matrix">
                  [{data.file_metadata.transform.join(", ")}]
                </code>
              </div>
            </section>
          )}

        </aside>

        {/* ── MAIN CANVAS ─────────────────────────────── */}
        <main className="canvas">
          {data ? (
            <>
              {/* Image Viewer */}
              <section className="card canvas-card">
                <div className="tab-bar">
                  {(["RAW","RGB","STRETCHED","MASK","UNET","CNN","VISION","RF"] as const).map(t => (
                    <button key={t} onClick={() => setTab(t)}
                      className={tab === t ? "tab active" : "tab"}>
                      {t === "RAW" ? "01 Raw " + (data.synth?.raw_label || "HH") : 
                       t === "RGB" ? "02 RGB Colour" :
                       t === "STRETCHED" ? "03 Composite" : 
                       t === "MASK" ? "04 K-Means" :
                       t === "UNET" ? "05 U-Net" :
                       t === "CNN" ? "06 CNN" : 
                       t === "VISION" ? "07 Vision" : "08 RF"}
                    </button>
                  ))}
                </div>
                <div className="image-stage">
                  <img key={tab} className="sar-img" alt={tab}
                    src={tab === "RAW" ? data.images.raw : 
                         tab === "RGB" ? data.images.rgb :
                         tab === "STRETCHED" ? data.images.stretched : 
                         tab === "MASK" ? data.images.mask :
                         tab === "UNET" ? data.images.unet :
                         tab === "CNN" ? data.images.cnn :
                         tab === "VISION" ? data.images.vision : data.images.rf} />
                  <div className="img-overlay">
                    <span>{data.file_metadata.resolution} px · {band}-Band</span>
                    <span>{tab === "RAW" ? "Linear amplitude stretch" : 
                           tab === "RGB" ? "RGB Colour Composite" :
                           tab === "STRETCHED" ? "10·log₁₀(σ) composite" : 
                           tab === "MASK" ? "Threshold K-means segmentation" :
                           tab === "UNET" ? "U-Net Semantic Segmentation" :
                           tab === "CNN" ? "FCN-CNN Semantic Segmentation" :
                           tab === "VISION" ? "Vision Transformer Segmentation" : "Random Forest Classifier"}</span>
                  </div>
                </div>
              </section>

              {/* Pipeline methodology */}
              <section className="card">
                <div className="card-header">PROCESSING PIPELINE · MATHEMATICAL METHODOLOGY</div>
                <div className="pipeline">
                  <PipelineStep n="01" title="Data Ingestion"
                    math="archive = zipfile.ZipFile(bytes) → TIF Extraction"
                    desc={"ZIP archive extracted in-memory via Python zipfile. Backend auto-detects and allocates the TIF tensor(s) to rasterio. Archive size: " + data.file_metadata.size_mb + " MB · dtype: " + data.file_metadata.dtype + "."} />
                  <PipelineStep n="02" title="Polarization Allocation"
                    math={data.synth?.method || "hv = hh × 0.3 + 𝒩(0, 10)"}
                    desc={(data.synth?.desc || "Missing polarization synthesized automatically.")} />
                  <PipelineStep n="03" title="Logarithmic dB Conversion"
                    math="σ_dB = 10 · log₁₀(σ°)  where σ° > 0"
                    desc={"Radar backscatter (σ°) is converted to decibel scale. Four channels are computed: HH_dB, HV_dB, Ratio_dB = 10·log₁₀(HH/HV), Diff_dB = 10·log₁₀(|HH−HV|). These are averaged to form the composite. dB range: " + data.db_stats.min + " to " + data.db_stats.max + " dB."} />
                  <PipelineStep n="04" title="Linear Stretch to 8-bit"
                    math={"stretched = clip((dB − " + data.band_config.lower_db + ") / (" + data.band_config.upper_db + " − " + data.band_config.lower_db + ") × 255, 0, 255)"}
                    desc={"Band-specific lower/upper bounds are applied to maximize dynamic contrast before rendering. These thresholds are chosen to preserve " + band + "-Band physics."} />
                  <PipelineStep n="05" title="Threshold Segmentation"
                    math={"Water: dB ≤ " + data.band_config.water_threshold_db + " | Land: ≤ " + data.band_config.land_threshold_db + " | Crop: ≤ " + data.band_config.crop_threshold_db + " | Mountain: > " + data.band_config.crop_threshold_db}
                    desc={"Each pixel of the composite dB image is assigned to one of four terrain classes using physics-derived thresholds calibrated for " + band + "-Band scattering behaviour."} />
                </div>
              </section>

              {/* Stats row */}
              <div className="stats-row">
                <section className="card stats-half">
                  <div className="card-header">RAW ARRAY STATISTICS · {data.synth?.raw_label || "HH"} POLARIZATION</div>
                  <div className="stat-table">
                    <StatRow label="Minimum"      value={data.raw_array_stats.min} unit="DN" />
                    <StatRow label="Maximum"      value={data.raw_array_stats.max} unit="DN" />
                    <StatRow label="Mean (μ)"     value={data.raw_array_stats.mean} unit="DN" />
                    <StatRow label="Std Dev (σ)"  value={data.raw_array_stats.std} unit="DN" />
                    <StatRow label="Variance (σ²)"value={data.raw_array_stats.variance} unit="DN²" />
                    <StatRow label="Median"       value={data.raw_array_stats.median} unit="DN" />
                    <StatRow label="5th Percentile" value={data.raw_array_stats.p5} unit="DN" />
                    <StatRow label="95th Percentile" value={data.raw_array_stats.p95} unit="DN" />
                  </div>
                </section>

                <section className="card stats-half">
                  <div className="card-header">COMPOSITE dB STATISTICS</div>
                  <div className="stat-table">
                    <StatRow label="Mean dB"      value={data.db_stats.mean} unit="dB" />
                    <StatRow label="Std Dev dB"   value={data.db_stats.std} unit="dB" />
                    <StatRow label="Min dB"       value={data.db_stats.min} unit="dB" />
                    <StatRow label="Max dB"       value={data.db_stats.max} unit="dB" />
                    <StatRow label="Dynamic Range" value={data.db_stats.range} unit="dB" />
                  </div>
                  <div className="card-header" style={{ marginTop: "1.5rem" }}>IMAGE QUALITY METRICS</div>
                  <div className="stat-table">
                    <StatRow label="Signal-to-Noise Ratio" value={data.quality_metrics.snr} unit="" />
                    <StatRow label="Entropy" value={data.quality_metrics.entropy_bits} unit="bits" />
                    <StatRow label="Dynamic Range" value={data.quality_metrics.dynamic_range_db} unit="DN" />
                    <StatRow label="Processing Time" value={data.quality_metrics.process_ms} unit="ms" />
                  </div>
                </section>
              </div>

            </>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <h2 className="empty-title">SAR Analytics Platform</h2>
              <p className="empty-desc">Select a GeoTIFF file and choose a SAR band to initialize the processing pipeline. All geospatial metadata, statistical analysis, and land-cover segmentation will appear here automatically.</p>
              <div className="empty-steps">
                <div className="step-item"><span className="step-num">01</span><span>Upload HH GeoTIFF</span></div>
                <div className="step-arrow">→</div>
                <div className="step-item"><span className="step-num">02</span><span>Select Band Model</span></div>
                <div className="step-arrow">→</div>
                <div className="step-item"><span className="step-num">03</span><span>Run Pipeline</span></div>
                <div className="step-arrow">→</div>
                <div className="step-item"><span className="step-num">04</span><span>Analyze Results</span></div>
              </div>
            </div>
          )}
        </main>

        {/* ── RIGHT SIDEBAR ────────────────────────────── */}
        <aside className="right-panel">

          {data ? (
            <>
              {/* Per-Model Classification + Metrics */}
              {(() => {
                const pmc = (data as any).per_model_classification;
                const models = [
                  { key: "kmeans", label: "K-Means Physics", icon: "◈", color: "#8e44ad",
                    acc: null, iou: null, f1: null, latency: null },
                  ...( data.model_metrics?.map(m => ({
                    key: m.Model === "U-Net" ? "unet" : m.Model === "CNN (FCN)" ? "cnn" : m.Model === "Vision Transformer" ? "vision" : "rf",
                    label: m.Model, icon: "◉", color: m.Model === "U-Net" ? "#2980b9" : m.Model === "CNN (FCN)" ? "#16a085" : m.Model === "Vision Transformer" ? "#d35400" : "#27ae60",
                    acc: m.Accuracy, iou: m.Mean_IoU, f1: m.F1_Score, latency: m.Latency_ms
                  })) || [] )
                ];
                return models.map(m => {
                  const cls = pmc?.[m.key];
                  return (
                    <section key={m.key} className="card" style={{ marginBottom: "8px" }}>
                      {/* Model Header */}
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "1px solid #1a1a1a" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                          <span style={{ color: m.color, fontSize: "14px" }}>{m.icon}</span>
                          <span style={{ fontSize: "11px", fontWeight: 600, color: "#e0e0e0", letterSpacing: "0.05em" }}>{m.label.toUpperCase()}</span>
                        </div>
                        {m.latency !== null && (
                          <span style={{ fontSize: "10px", color: "#e67e22", fontFamily: "monospace", background: "#1a1000", padding: "2px 6px", borderRadius: "3px" }}>{m.latency}ms</span>
                        )}
                      </div>

                      {/* Accuracy row (not for K-Means) */}
                      {m.acc !== null && (
                        <div style={{ display: "flex", justifyContent: "space-around", padding: "8px 0", borderBottom: "1px solid #111" }}>
                          {[["Acc", m.acc], ["IoU", m.iou], ["F1", m.f1]].map(([label, val]) => (
                            <div key={String(label)} style={{ textAlign: "center" }}>
                              <div style={{ fontSize: "13px", fontFamily: "monospace", color: "#fff", fontWeight: 600 }}>{(val as number).toFixed(3)}</div>
                              <div style={{ fontSize: "9px", color: "#555", marginTop: "2px", letterSpacing: "0.06em" }}>{label}</div>
                            </div>
                          ))}
                        </div>
                      )}
                      {m.acc === null && (
                        <div style={{ padding: "6px 14px 2px", fontSize: "9px", color: "#555", letterSpacing: "0.05em" }}>THRESHOLD-BASED · NO TRAINING REQUIRED</div>
                      )}

                      {/* Land Cover Breakdown */}
                      {cls && (
                        <div style={{ padding: "8px 14px 10px" }}>
                          {CLASSES.map(c => (
                            <div key={c.key} style={{ marginBottom: "6px" }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: c.color, flexShrink: 0 }}></div>
                                  <span style={{ fontSize: "10px", color: "#aaa" }}>{c.label}</span>
                                </div>
                                <span style={{ fontSize: "10px", fontFamily: "monospace", color: "#e0e0e0" }}>{cls[c.pctKey]}%</span>
                              </div>
                              <div style={{ height: "3px", background: "#1a1a1a", borderRadius: "2px" }}>
                                <div style={{ height: "3px", borderRadius: "2px", background: c.color, width: cls[c.pctKey] + "%", transition: "width 0.5s ease" }}></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </section>
                  );
                });
              })()}


              {/* dB Histogram */}
              <section className="card chart-card">
                <div className="card-header">COMPOSITE dB DISTRIBUTION</div>
                <p className="chart-formula">σ_composite(dB) histogram · {data.histogram.length} bins</p>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data.histogram} margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor="#ffffff" stopOpacity={0.15}/>
                          <stop offset="95%" stopColor="#ffffff" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.05)" vertical={false}/>
                      <XAxis dataKey="bin" tick={{ fill: "#555", fontSize: 9 }} tickLine={false} axisLine={false} minTickGap={30}/>
                      <YAxis tick={{ fill: "#555", fontSize: 9 }} tickLine={false} axisLine={false}
                        tickFormatter={v => v >= 1000 ? (v/1000).toFixed(0)+"k" : String(v)}/>
                      <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: 4, fontSize: 12 }}
                        itemStyle={{ color: "#fff" }} labelStyle={{ color: "#888" }}/>
                      <Area type="monotone" dataKey="count" stroke="#ffffff" strokeWidth={1.5}
                        fillOpacity={1} fill="url(#areaGrad)"/>
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </section>

              {/* Raw histogram */}
              <section className="card chart-card">
                <div className="card-header">RAW AMPLITUDE DISTRIBUTION</div>
                <p className="chart-formula">DN value histogram · non-zero pixels</p>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.raw_histogram} margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.05)" vertical={false}/>
                      <XAxis dataKey="bin" tick={{ fill: "#555", fontSize: 9 }} tickLine={false} axisLine={false} minTickGap={30}/>
                      <YAxis tick={{ fill: "#555", fontSize: 9 }} tickLine={false} axisLine={false}
                        tickFormatter={v => v >= 1000 ? (v/1000).toFixed(0)+"k" : String(v)}/>
                      <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: 4, fontSize: 12 }}
                        itemStyle={{ color: "#fff" }} labelStyle={{ color: "#888" }}/>
                      <Bar dataKey="count" fill="#3a3a3a" radius={[2,2,0,0]}>
                        {data.raw_histogram.map((_, i) => <Cell key={i} fill={i % 2 === 0 ? "#4a4a4a" : "#333"}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>

            </>
          ) : (
            <div className="right-empty">
              <p>Charts and classification data will appear here after processing.</p>
            </div>
          )}

        </aside>
      </div>

      <style dangerouslySetInnerHTML={{ __html: CSS }} />
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────
function KV({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="kv-row">
      <span className="kv-key">{k}</span>
      <span className={mono ? "kv-val mono" : "kv-val"}>{v}</span>
    </div>
  );
}

function StatRow({ label, value, unit }: { label: string; value: number; unit: string }) {
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : value} <span className="stat-unit">{unit}</span></span>
    </div>
  );
}

function PipelineStep({ n, title, math, desc }: { n: string; title: string; math: string; desc: string }) {
  return (
    <div className="pipeline-step">
      <div className="pipeline-num">{n}</div>
      <div className="pipeline-body">
        <div className="pipeline-title">{title}</div>
        <code className="pipeline-math">{math}</code>
        <p className="pipeline-desc">{desc}</p>
      </div>
    </div>
  );
}

// ── CSS ───────────────────────────────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.root {
  background: #000;
  color: #e8e8e8;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  line-height: 1.5;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* NAV */
.nav {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0 24px; height: 52px;
  border-bottom: 1px solid #1e1e1e;
  background: #000;
  position: sticky; top: 0; z-index: 100;
}
.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-logo { font-size: 18px; color: #fff; }
.nav-title { font-size: 14px; font-weight: 600; color: #fff; letter-spacing: -0.01em; }
.nav-sep { color: #333; }
.nav-sub { font-size: 12px; color: #555; }
.nav-right { display: flex; align-items: center; gap: 16px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e88; }
.status-txt { font-size: 12px; color: #555; }

.band-toggle { display: flex; border: 1px solid #222; border-radius: 6px; overflow: hidden; }
.band-btn {
  padding: 5px 18px; background: transparent; border: none; border-right: 1px solid #222;
  font-size: 12px; font-weight: 500; color: #666; cursor: pointer; transition: all .15s;
}
.band-btn:last-child { border-right: none; }
.band-btn.active { background: #fff; color: #000; }
.band-btn:hover:not(.active) { background: #111; color: #ccc; }

/* BODY */
.body {
  display: grid;
  grid-template-columns: 320px 1fr 320px;
  gap: 0;
  flex: 1;
  overflow: hidden;
}

/* CARDS */
.card {
  background: #0a0a0a;
  border: 1px solid #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
  flex-shrink: 0;
}
.card-header {
  padding: 10px 16px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
  color: #555; text-transform: uppercase;
  border-bottom: 1px solid #1a1a1a;
  background: #050505;
}

/* SIDEBAR */
.sidebar {
  border-right: 1px solid #1a1a1a;
  padding: 16px;
  display: flex; flex-direction: column; gap: 12px;
  overflow-y: auto; height: calc(100vh - 52px);
}
.sidebar::-webkit-scrollbar { width: 4px; }
.sidebar::-webkit-scrollbar-track { background: #000; }
.sidebar::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }

/* UPLOAD */
.upload-zone {
  padding: 24px 16px;
  display: flex; flex-direction: column; align-items: center;
  cursor: pointer; border-bottom: 1px solid #1a1a1a;
  transition: background .15s;
}
.upload-zone:hover { background: #0f0f0f; }
.upload-icon { font-size: 28px; color: #444; margin-bottom: 10px; }
.upload-name { font-size: 12px; font-weight: 500; color: #ccc; text-align: center; word-break: break-all; }
.upload-hint { font-size: 11px; color: #444; margin-top: 4px; text-align: center; }

.run-btn {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  width: 100%; margin: 12px 0 0;
  padding: 10px; border: none; border-radius: 6px;
  background: #fff; color: #000;
  font-size: 13px; font-weight: 600; cursor: pointer; transition: all .2s;
}
.run-btn:hover:not(:disabled) { background: #e5e5e5; }
.run-btn:disabled { opacity: .45; cursor: not-allowed; }
.run-btn.running { background: #111; color: #666; border: 1px solid #222; }
.reset-btn {
  width: 100%; margin-top: 8px; padding: 8px;
  background: transparent; border: 1px solid #222; border-radius: 6px;
  color: #555; font-size: 12px; cursor: pointer;
}
.reset-btn:hover { border-color: #444; color: #aaa; }
.error-box { margin-top: 10px; padding: 10px; background: #1a0808; border: 1px solid #3a1010; border-radius: 6px; color: #f87171; font-size: 12px; }

/* KV */
.kv-grid { display: flex; flex-direction: column; padding: 8px 16px; gap: 2px; }
.kv-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #111; gap: 12px; }
.kv-row:last-child { border-bottom: none; }
.kv-key { color: #555; font-size: 11px; flex-shrink: 0; }
.kv-val { color: #ccc; font-size: 11px; text-align: right; word-break: break-all; }
.kv-val.mono { font-family: 'JetBrains Mono', monospace; font-size: 10px; }

.info-block { padding: 8px 16px; border-top: 1px solid #111; }
.info-label { font-size: 10px; color: #444; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.info-val { font-size: 12px; color: #bbb; }

.transform-matrix {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; color: #888;
  word-break: break-all; display: block; margin-top: 6px;
}

/* CANVAS */
.canvas {
  padding: 16px;
  display: flex; flex-direction: column; gap: 12px;
  overflow-y: auto; height: calc(100vh - 52px);
}
.canvas::-webkit-scrollbar { width: 4px; }
.canvas::-webkit-scrollbar-track { background: #000; }
.canvas::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }

.canvas-card { display: flex; flex-direction: column; }

.tab-bar { display: flex; border-bottom: 1px solid #1a1a1a; }
.tab {
  flex: 1; padding: 10px 8px;
  background: transparent; border: none; border-bottom: 2px solid transparent;
  font-size: 11px; font-weight: 500; color: #555; cursor: pointer; transition: all .15s;
  letter-spacing: 0.02em;
}
.tab.active { color: #fff; border-bottom-color: #fff; }
.tab:hover:not(.active) { color: #aaa; }

.image-stage { position: relative; background: #050505; aspect-ratio: 1 / .7; }
.sar-img { width: 100%; height: 100%; object-fit: contain; display: block; animation: fadeImg .5s ease-out; }
@keyframes fadeImg { from { opacity: 0; } to { opacity: 1; } }

.img-overlay {
  position: absolute; bottom: 12px; left: 12px; right: 12px;
  display: flex; justify-content: space-between;
  background: rgba(0,0,0,.7); backdrop-filter: blur(6px);
  border: 1px solid #222; border-radius: 4px;
  padding: 6px 12px; font-size: 11px; color: #888;
  font-family: 'JetBrains Mono', monospace;
}

/* PIPELINE */
.pipeline { display: flex; flex-direction: column; }
.pipeline-step { display: flex; gap: 16px; padding: 16px; border-bottom: 1px solid #111; }
.pipeline-step:last-child { border-bottom: none; }
.pipeline-num {
  flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
  border: 1px solid #333; display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 600; color: #666; margin-top: 2px;
}
.pipeline-body { display: flex; flex-direction: column; gap: 6px; }
.pipeline-title { font-size: 13px; font-weight: 600; color: #e8e8e8; }
.pipeline-math {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #aaa;
  background: #0f0f0f; border: 1px solid #1e1e1e;
  padding: 6px 10px; border-radius: 4px;
  display: block; word-break: break-all;
}
.pipeline-desc { font-size: 12px; color: #666; line-height: 1.6; }

/* STATS ROW */
.stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.stats-half { display: flex; flex-direction: column; }
.stat-table { display: flex; flex-direction: column; padding: 8px 16px; }
.stat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 0; border-bottom: 1px solid #111;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: #666; font-size: 11px; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #e8e8e8; }
.stat-unit { font-size: 10px; color: #555; }

/* RIGHT PANEL */
.right-panel {
  border-left: 1px solid #1a1a1a;
  padding: 16px;
  display: flex; flex-direction: column; gap: 12px;
  overflow-y: auto; height: calc(100vh - 52px);
}
.right-panel::-webkit-scrollbar { width: 4px; }
.right-panel::-webkit-scrollbar-track { background: #000; }
.right-panel::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
.right-empty { color: #333; font-size: 12px; padding: 24px; text-align: center; }

/* CLASSIFICATION */
.class-list { display: flex; flex-direction: column; padding: 12px 16px; gap: 16px; }
.class-item { display: flex; flex-direction: column; gap: 5px; }
.class-top { display: flex; align-items: center; gap: 8px; }
.class-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.class-label { font-size: 12px; color: #aaa; flex: 1; }
.class-pct { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #fff; }
.bar-bg { width: 100%; height: 4px; background: #1a1a1a; border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; transition: width 1.2s cubic-bezier(.16,1,.3,1); }
.class-px { font-size: 10px; color: #444; font-family: 'JetBrains Mono', monospace; }

/* CHARTS */
.chart-card { display: flex; flex-direction: column; }
.chart-formula { padding: 6px 16px 0; font-size: 10px; color: #444; font-family: 'JetBrains Mono', monospace; }
.chart-wrap { height: 160px; padding: 8px 8px 8px 8px; }

/* EMPTY */
.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; padding: 48px; text-align: center; gap: 20px;
}
.empty-icon { font-size: 40px; color: #222; }
.empty-title { font-size: 20px; font-weight: 600; color: #fff; letter-spacing: -0.02em; }
.empty-desc { font-size: 13px; color: #555; max-width: 480px; line-height: 1.7; }
.empty-steps { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
.step-item { display: flex; align-items: center; gap: 8px; }
.step-num {
  width: 24px; height: 24px; border-radius: 50%; border: 1px solid #333;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 600; color: #666;
}
.step-arrow { color: #333; font-size: 18px; }

/* SPINNER */
.spinner {
  width: 14px; height: 14px;
  border: 2px solid rgba(0,0,0,.3);
  border-top-color: #000;
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
`;
