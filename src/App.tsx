import React, { useState, useEffect, useRef } from 'react';
import { AudioEngine } from './services/AudioEngine';
import { generateVoicePersona } from './services/geminiService';
import { BasinService } from './services/BasinService';
import { FMJService } from './services/FMJService';
import { CRISPRService } from './services/CRISPRService';
import { DirectorService } from './services/DirectorService';
import { NeuralSuitService } from './services/NeuralSuitService';
import {
  AudioSettings,
  Preset,
  PluginFormat,
  ReactionStep,
  LoopLayer,
  HarmonizerVoice,
  BasinState,
  FMJState,
  CRISPRState,
  DirectorState,
  NeuralSuitState
} from './types';
import { DEFAULT_SETTINGS, PRESETS } from './constants';
import Visualizer from './components/Visualizer';
import BasinModule from './components/BasinModule';
import Control, { ControlGroup, Knob } from './components/ControlGroup';
import {
  Activity,
  Mic,
  Zap,
  Waves,
  Cpu,
  Sparkles,
  RefreshCw,
  Power,
  Monitor,
  Database,
  Atom,
  Play,
  Square as SquareIcon,
  Plus,
  Trash2,
  Clock,
  Volume2,
  VolumeX,
  Disc,
  Combine,
  Tally4,
  Flame,
  Layers,
  Save,
  X,
  Music,
  UserCheck,
  Bot,
  Box,
  MoveHorizontal,
  Maximize,
  HelpCircle,
  Info,
  ChevronRight,
  BookOpen,
  Terminal,
  Code,
  Download
} from 'lucide-react';

const LEDBar: React.FC<{ value: number; label: string }> = ({ value, label }) => {
  const bars = 10;
  const active = Math.floor(value * bars * 3);
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex flex-col-reverse gap-[2px] bg-black/80 p-1 border border-white/5 rounded">
        {Array.from({ length: bars }).map((_, i) => (
          <div
            key={i}
            className={`led-segment ${i < active ? (i > 8 ? 'bg-red-500 shadow-[0_0_5px_red]' : (i > 6 ? 'bg-yellow-400' : 'bg-green-500')) : ''}`}
          />
        ))}
      </div>
      <span className="text-[7px] font-bold text-slate-600 uppercase">{label}</span>
    </div>
  );
};

const WaveformSymbol: React.FC<{ type: string; active: boolean }> = ({ type, active }) => {
  const color = active ? "#000" : "#00f2ff";

  switch (type) {
    case 'sawtooth':
      return (
        <svg width="20" height="10" viewBox="0 0 20 10" className="opacity-80">
          <path d="M 0 10 L 10 0 L 10 10 L 20 0 L 20 10" fill="none" stroke={color} strokeWidth="1.5" />
        </svg>
      );
    case 'square':
      return (
        <svg width="20" height="10" viewBox="0 0 20 10" className="opacity-80">
          <path d="M 0 10 L 0 0 L 10 0 L 10 10 L 20 10 L 20 0 L 20 10" fill="none" stroke={color} strokeWidth="1.5" />
        </svg>
      );
    case 'pulse':
      return (
        <svg width="20" height="10" viewBox="0 0 20 10" className="opacity-80">
          <path d="M 0 10 L 4 10 L 4 0 L 8 0 L 8 10 L 20 10" fill="none" stroke={color} strokeWidth="1.5" />
        </svg>
      );
    case 'noise':
      return (
        <svg width="20" height="10" viewBox="0 0 20 10" className="opacity-80">
          <path d="M 0 5 L 2 2 L 4 8 L 6 3 L 8 9 L 10 1 L 12 7 L 14 4 L 16 10 L 18 2 L 20 5" fill="none" stroke={color} strokeWidth="1" />
        </svg>
      );
    default:
      return null;
  }
};

const HelpOverlay: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const sections = [
    {
      title: "Neural Engine",
      icon: <Zap size={18} className="text-[#bf00ff]" />,
      desc: "Powered by Gemini, this module transforms text prompts into DSP configurations. It understands descriptive language like 'Demonic whisper' or 'Cyberpunk broadcast' and maps them to granular and vocoder parameters."
    },
    {
      title: "Jam Engine",
      icon: <Bot size={18} className="text-cyan-400" />,
      desc: "Agentic backup singers that autonomously improvise based on your input. 'Chaos' controls the frequency of their decisions, while 'Style' defines their musical relationship to your voice."
    },
    {
      title: "Formant Geometry",
      icon: <Waves size={18} className="text-emerald-500" />,
      desc: "Spectral envelope manipulation. Shifting formants changes the perceived size of the vocal tract without altering pitch. Bandwidth (Q) sharpens or blurs the resonance of these spectral peaks."
    },
    {
      title: "Granular Engine",
      icon: <Sparkles size={18} className="text-orange-500" />,
      desc: "Deconstructs audio into tiny 'grains'. 'Density' controls overlapping grains per second, while 'Size' determines duration. Modulating grain pitch creates ethereal textures or mechanical stuttering."
    },
    {
      title: "3D Synthesis",
      icon: <Maximize size={18} className="text-blue-500" />,
      desc: "Immersive spatial processor. Width (X) handles stereo spread, Depth (Y) controls an algorithmic feedback delay network, and Dimension (Z) adds a temporal-chorus thickening effect."
    },
    {
      title: "Native Bridge (Xcode)",
      icon: <Code size={18} className="text-[#ccff00]" />,
      desc: "To wrap this for GarageBand (macOS): 1. Create an AUv3 App Extension in Xcode. 2. Embed a WKWebView. 3. Set 'NSMicrophoneUsageDescription' in Info.plist. 4. Use 'WKWebsiteDataStore' to allow Gemini API persistence."
    },
    {
      title: "Chain Lab",
      icon: <Layers size={18} className="text-[#ffaa00]" />,
      desc: "A macro-sequencer for the entire machine. Capture 'Nodes' (snapshots) and trigger them sequentially to create evolving soundscapes or automated performance variations."
    },
    {
      title: "DAW Integration",
      icon: <Terminal size={18} className="text-[#ccff00]" />,
      desc: "To use VoxGrain in GarageBand or Logic: Use a virtual cable (like BlackHole). Route your browser audio to BlackHole, then set your DAW's input to the same cable to record processed vocals in real-time."
    }
  ];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-6 backdrop-blur-md bg-black/80 animate-in fade-in zoom-in duration-300">
      <div className="w-full max-w-4xl bg-[#121418] border border-white/10 rounded-xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
        <div className="p-4 bg-black/40 border-b border-white/5 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <BookOpen className="text-[#ccff00]" size={20} />
            <h2 className="font-orbitron text-xs font-black uppercase tracking-[0.4em] text-white">Operational Manual</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full transition-colors text-slate-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {sections.map((s, idx) => (
              <div key={idx} className="p-4 bg-white/5 border border-white/5 rounded-lg hover:border-white/20 transition-all group">
                <div className="flex items-center gap-3 mb-3">
                  <div className="p-2 bg-black/40 rounded-lg group-hover:scale-110 transition-transform">{s.icon}</div>
                  <h3 className="font-orbitron text-[10px] font-black uppercase text-white tracking-widest">{s.title}</h3>
                </div>
                <p className="text-[11px] leading-relaxed text-slate-400 font-medium">{s.desc}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 p-4 bg-[#ccff00]/5 border border-[#ccff00]/10 rounded-lg">
            <h4 className="text-[10px] font-black uppercase text-[#ccff00] mb-2 flex items-center gap-2">
              <Info size={12} /> Pro Tip: Morphing
            </h4>
            <p className="text-[10px] text-slate-400 italic">
              Try combining the Granular Engine with the 3D Dimension control while the Jam Engine is active.
              The resulting interplay between agentic improvisation and temporal grain manipulation creates
              complex, shifting textures impossible with standard vocoders.
            </p>
          </div>
        </div>

        <div className="p-4 bg-black/40 border-t border-white/5 text-center">
          <button
            onClick={onClose}
            className="px-8 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-[10px] font-black uppercase tracking-[0.2em] text-white transition-all"
          >
            Acknowledge & Close
          </button>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [settings, setSettings] = useState<AudioSettings>(DEFAULT_SETTINGS);
  const [format, setFormat] = useState<PluginFormat>('VST3');
  const [isActive, setIsActive] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [levels, setLevels] = useState({ input: 0, output: 0 });
  const [agentActivity, setAgentActivity] = useState<{ freq: number, gain: number }[]>([]);
  const [isPersonaLoading, setIsPersonaLoading] = useState(false);
  const [personaPrompt, setPersonaPrompt] = useState('');
  const [analyzer, setAnalyzer] = useState<AnalyserNode | null>(null);
  const [userPresets, setUserPresets] = useState<Preset[]>(() => {
    const saved = localStorage.getItem('voxgrain_user_presets');
    return saved ? JSON.parse(saved) : [];
  });
  const [newPresetName, setNewPresetName] = useState('');

  // Looper State
  const [loops, setLoops] = useState<LoopLayer[]>([]);
  const [isRecordingLoop, setIsRecordingLoop] = useState(false);

  // Basin & Module State
  const [basinState, setBasinState] = useState<BasinState>({ hubs: [], clouds: [], fermentationProgress: 0, routingActive: false });
  const [fmjState, setFmjState] = useState<FMJState>({ enabled: false, integrity: 1, hardness: 0.5, thrust: 0.2, hazardIntensity: 0 });
  const [crisprState, setCrisprState] = useState<CRISPRState>({ enabled: false, sequence: '', patchProgress: 0, isAnalyzing: false });
  const [directorState, setDirectorState] = useState<DirectorState>({ enabled: false, mode: 'WITNESSING', resonance: 0.5, tone: 'clarity' });
  const [neuralSuitState, setNeuralSuitState] = useState<NeuralSuitState>({ enabled: false, integrity: 100, swarmLogic: 'IDLE', hazards: [] });

  // Reaction Chain State
  const [chain, setChain] = useState<ReactionStep[]>([]);
  const [isPlayingChain, setIsPlayingChain] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(-1);
  const [chainLoop, setChainLoop] = useState(true);

  const engineRef = useRef<AudioEngine | null>(null);
  const basinServiceRef = useRef(new BasinService());
  const fmjServiceRef = useRef(new FMJService());
  const crisprServiceRef = useRef(new CRISPRService());
  const directorServiceRef = useRef(new DirectorService());
  const neuralSuitServiceRef = useRef(new NeuralSuitService());
  const chainTimerRef = useRef<any>(null);

  useEffect(() => {
    localStorage.setItem('voxgrain_user_presets', JSON.stringify(userPresets));
  }, [userPresets]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (engineRef.current && isActive) {
        const currentLevels = engineRef.current.getLevels();
        setLevels(currentLevels);

        if (settings.jamEnabled) {
          setAgentActivity(engineRef.current.getAgentActivity());
        }

        // Update Basin & Sub-modules
        if (settings.basinEnabled) {
           setBasinState(basinServiceRef.current.update(settings.basinEnabled));
        }

        if (settings.fmjEnabled) {
           const fState = fmjServiceRef.current.update(currentLevels.output, settings.fmjEnabled);
           setFmjState({ ...fState });
        }

        if (settings.crisprEnabled) {
           setCrisprState({ ...crisprServiceRef.current.update(settings.crisprEnabled) });
        }

        if (settings.directorEnabled) {
           setDirectorState({ ...directorServiceRef.current.update(settings.directorEnabled, currentLevels.output) });
        }

        if (settings.neuralSuitEnabled) {
           const nsState = neuralSuitServiceRef.current.update(settings.neuralSuitEnabled, currentLevels.output > 0.7);
           setNeuralSuitState({ ...nsState });
           engineRef.current.updateIntegrityEffects(nsState.integrity);
        }
      }
    }, 40);
    return () => clearInterval(interval);
  }, [isActive, settings.jamEnabled, settings.basinEnabled, settings.fmjEnabled, settings.crisprEnabled, settings.directorEnabled, settings.neuralSuitEnabled]);

  const startEngine = async () => {
    if (!engineRef.current) {
      engineRef.current = new AudioEngine();
      await engineRef.current.startMic();
      setAnalyzer(engineRef.current.getAnalyzer());
    }
    engineRef.current.updateSettings(settings);
    setIsActive(true);
  };

  const updateSetting = <K extends keyof AudioSettings>(key: K, value: AudioSettings[K]) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    if (engineRef.current) engineRef.current.updateSettings(newSettings);
  };

  const updateHarmonizerVoice = (index: number, updates: Partial<HarmonizerVoice>) => {
    const newVoices = [...settings.harmonizerVoices];
    newVoices[index] = { ...newVoices[index], ...updates };
    updateSetting('harmonizerVoices', newVoices);
  };

  const handlePersonaGen = async () => {
    if (!personaPrompt) return;
    setIsPersonaLoading(true);
    try {
      const result = await generateVoicePersona(personaPrompt);
      const newSettings = { ...settings, ...result.settings };
      setSettings(newSettings);
      if (engineRef.current) engineRef.current.updateSettings(newSettings);
    } catch (e) { console.error(e); } finally { setIsPersonaLoading(false); }
  };

  const toggleRecording = () => {
    if (!isActive) return;
    if (isRecordingLoop) {
      const id = engineRef.current?.stopLoopRecording();
      if (id) {
        setLoops([...loops, { id, timestamp: Date.now(), duration: 0, volume: 1.0, isMuted: false }]);
      }
      setIsRecordingLoop(false);
    } else {
      engineRef.current?.startLoopRecording();
      setIsRecordingLoop(true);
    }
  };

  const removeLoop = (id: string) => {
    engineRef.current?.removeLayer(id);
    setLoops(loops.filter(l => l.id !== id));
  };

  const toggleMute = (id: string) => {
    engineRef.current?.toggleLayerMute(id);
    setLoops(loops.map(l => l.id === id ? { ...l, isMuted: !l.isMuted } : l));
  };

  const captureNode = () => {
    const step: ReactionStep = {
      id: Math.random().toString(36).substr(2, 9),
      settings: { ...settings },
      duration: 1500,
      morph: true
    };
    setChain([...chain, step]);
  };

  const clearChain = () => {
    stopChain();
    setChain([]);
  };

  const stopChain = () => {
    setIsPlayingChain(false);
    setActiveStepIndex(-1);
    if (chainTimerRef.current) clearTimeout(chainTimerRef.current);
  };

  const triggerChain = () => {
    if (chain.length < 1) return;
    setIsPlayingChain(true);
    playStep(0);
  };

  const playStep = (index: number) => {
    if (index >= chain.length) {
      if (chainLoop) {
        playStep(0);
      } else {
        stopChain();
      }
      return;
    }

    setActiveStepIndex(index);
    const step = chain[index];
    setSettings(step.settings);
    if (engineRef.current) engineRef.current.updateSettings(step.settings);

    chainTimerRef.current = setTimeout(() => {
      playStep(index + 1);
    }, step.duration);
  };

  const savePreset = () => {
    if (!newPresetName.trim()) return;
    const newPreset: Preset = {
      name: newPresetName.trim(),
      description: "User defined preset",
      settings: JSON.parse(JSON.stringify(settings))
    };
    setUserPresets([...userPresets, newPreset]);
    setNewPresetName('');
  };

  const deleteUserPreset = (name: string) => {
    setUserPresets(userPresets.filter(p => p.name !== name));
  };

  const toggleBasinModule = (modName: string) => {
    const key = `${modName}Enabled` as keyof AudioSettings;
    updateSetting(key, !settings[key]);
  };

  const handleExportManifest = () => {
    const manifest = {
      app: "VoxGrain_X",
      version: "2.6.0",
      target: "Xcode AUv3",
      permissions: ["microphone", "network"],
      plist: {
        NSMicrophoneUsageDescription: "VoxGrain requires microphone access to process Neural Vocoding in real-time.",
        AppTransportSecurity: "Allows Gemini API cloud connection."
      }
    };
    console.log("Developer Manifest Generated:", manifest);
    alert("Developer Manifest generated in console. Use this for your Xcode AUv3 setup.");
  };

  const skinClass = `skin-${format.toLowerCase()}`;

  return (
    <div className={`min-h-screen p-4 flex items-center justify-center transition-colors duration-500 ${skinClass}`}>
      <HelpOverlay isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />

      <div className="chassis w-full max-w-[1280px] rounded-lg overflow-hidden flex flex-col relative shadow-2xl">

        <div className="bg-black/90 p-2 flex justify-between items-center border-b border-white/5 z-20">
           <div className="flex gap-4 pl-4 items-center">
              {(['VST3', 'AU', 'AAX', 'STANDALONE'] as PluginFormat[]).map(f => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`text-[9px] font-black uppercase tracking-widest px-3 py-1 rounded transition-all ${format === f ? 'bg-white/10 text-[#00f2ff]' : 'text-slate-600 hover:text-white'}`}
                >
                  {f}
                </button>
              ))}
              <div className="w-[1px] h-4 bg-white/5 mx-2" />
              <button
                onClick={() => setIsHelpOpen(true)}
                className="flex items-center gap-2 text-[9px] font-black uppercase tracking-widest text-slate-500 hover:text-[#ccff00] transition-colors"
              >
                <HelpCircle size={14} /> Help
              </button>
              <button
                onClick={handleExportManifest}
                className="flex items-center gap-2 text-[9px] font-black uppercase tracking-widest text-slate-500 hover:text-cyan-400 transition-colors"
                title="Generate Xcode AUv3 Manifest"
              >
                <Download size={14} /> Bundle
              </button>
           </div>
           <div className="flex items-center gap-6 pr-4">
              <div className="flex gap-2">
                 <LEDBar value={levels.input} label="IN" />
                 <LEDBar value={levels.output} label="OUT" />
              </div>
              <button onClick={startEngine} className={`w-8 h-8 rounded-full border-2 flex items-center justify-center transition-all ${isActive ? 'bg-[#ccff00] border-white shadow-[0_0_15px_#ccff00]' : 'border-slate-800 bg-black text-slate-800'}`}>
                <Power size={14} className={isActive ? 'text-black' : ''} />
              </button>
           </div>
        </div>

        <div className="p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 brushed bg-[#1a1c21]">

           <div className="lg:col-span-8 flex flex-col gap-4">

              <div className="rack-module lcd-display p-3 lcd-glow relative border-2 border-black">
                <div className="absolute top-2 right-4 text-[7px] text-cyan-500/30 font-mono flex items-center gap-2">
                   <Activity size={10} /> RT_MONITOR: ACTIVE
                </div>
                <Visualizer analyzer={analyzer} />
              </div>

              {/* JAM ENGINE RACK */}
              <div className="rack-module p-5 border-l-4 border-cyan-400 bg-black/40 relative">
                 <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                       <Bot size={20} className={`text-cyan-400 ${settings.jamEnabled ? 'animate-pulse' : ''}`} />
                       <div>
                          <h3 className="text-[10px] font-black font-orbitron tracking-[0.2em] text-slate-400 uppercase">Jam Engine / Agentic Voices</h3>
                          <p className="text-[8px] font-mono text-slate-600 uppercase">Autonomous Improvisation Node</p>
                       </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex gap-1">
                        {(['chorus', 'duo', 'dissonant', 'swarm'] as const).map(style => (
                          <button
                            key={style}
                            onClick={() => updateSetting('jamStyle', style)}
                            className={`px-2 py-1 text-[7px] font-black uppercase rounded border transition-all ${settings.jamStyle === style ? 'bg-cyan-500 text-black border-white' : 'bg-black/40 text-slate-600 border-white/5'}`}
                          >
                            {style}
                          </button>
                        ))}
                      </div>
                      <div
                        onClick={() => updateSetting('jamEnabled', !settings.jamEnabled)}
                        className={`w-12 h-6 rounded-full p-1 cursor-pointer transition-all ${settings.jamEnabled ? 'bg-cyan-400 shadow-[0_0_10px_#22d3ee]' : 'bg-slate-800'}`}
                      >
                        <div className={`w-4 h-4 bg-white rounded-full transition-all ${settings.jamEnabled ? 'translate-x-6' : 'translate-x-0'}`} />
                      </div>
                    </div>
                 </div>

                 <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                    <div className="md:col-span-4 flex flex-wrap gap-6">
                       <Knob
                         label="Chaos"
                         value={settings.jamChaos}
                         min={0}
                         max={1}
                         step={0.01}
                         color="#22d3ee"
                         onChange={(v) => updateSetting('jamChaos', v)}
                       />
                       <Knob
                         label="Singers"
                         value={settings.jamSingerCount}
                         min={1}
                         max={4}
                         step={1}
                         color="#22d3ee"
                         onChange={(v) => updateSetting('jamSingerCount', v)}
                       />
                    </div>
                    <div className="md:col-span-8 bg-black/60 rounded p-4 border border-white/5 flex gap-4 h-24 items-end justify-around">
                       {Array.from({ length: 4 }).map((_, i) => {
                          const isActive = i < settings.jamSingerCount && settings.jamEnabled;
                          const activity = agentActivity[i] || { freq: 440, gain: 0 };
                          const height = isActive ? (activity.gain * 100) : 0;
                          return (
                            <div key={i} className="flex flex-col items-center gap-2 flex-1">
                               <div className="w-full bg-slate-900 rounded-t-sm relative h-12 overflow-hidden border-b border-white/10">
                                  <div
                                    className="absolute bottom-0 left-0 right-0 bg-cyan-400/50 transition-all duration-300"
                                    style={{ height: `${height}%`, boxShadow: `0 0 10px rgba(34,211,238,${activity.gain})` }}
                                  />
                               </div>
                               <span className={`text-[6px] font-black uppercase ${isActive ? 'text-cyan-400' : 'text-slate-800'}`}>Singer {i+1}</span>
                            </div>
                          );
                       })}
                    </div>
                 </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                 <ControlGroup title="Formant Geometry" accentClass="border-emerald-500">
                    <div className="flex flex-wrap gap-8">
                       <Knob
                         label="F-Shift"
                         value={settings.formantShift}
                         min={-2}
                         max={2}
                         step={0.01}
                         unit=""
                         color="#10b981"
                         onChange={(v) => updateSetting('formantShift', v)}
                       />
                       <Knob
                         label="Bandwidth"
                         value={settings.formantBandwidth}
                         min={0.5}
                         max={20}
                         step={0.1}
                         unit="Q"
                         color="#10b981"
                         onChange={(v) => updateSetting('formantBandwidth', v)}
                       />
                    </div>
                 </ControlGroup>

                 <ControlGroup title="Vocoder Core" accentClass="border-cyan-500">
                    <div className="flex flex-col gap-4 w-full">
                      <div className="flex flex-wrap gap-6">
                        <Knob label="Carrier" value={settings.carrierFreq} min={40} max={600} step={1} unit="Hz" color="#00f2ff" onChange={(v) => updateSetting('carrierFreq', v)} />
                        <Knob label="Bands" value={settings.vocoderBands} min={4} max={32} step={1} color="#00f2ff" onChange={(v) => updateSetting('vocoderBands', v)} />
                        <Knob label="Decay" value={settings.vocoderDecay} min={0} max={1} step={0.01} color="#00f2ff" onChange={(v) => updateSetting('vocoderDecay', v)} />
                      </div>

                      <div className="flex flex-col gap-1.5 pt-1">
                        <span className="text-[7px] font-black text-slate-600 uppercase tracking-widest pl-1">Carrier Waveform</span>
                        <div className="grid grid-cols-4 gap-1">
                           {(['sawtooth', 'square', 'pulse', 'noise'] as const).map(type => (
                              <button
                                key={type}
                                onClick={() => updateSetting('carrierType', type)}
                                className={`flex flex-col items-center justify-center gap-1.5 py-2 px-1 rounded border transition-all ${settings.carrierType === type ? 'bg-[#00f2ff] text-black border-white shadow-[0_0_10px_rgba(0,242,255,0.3)]' : 'bg-black/40 text-slate-600 border-white/5 hover:border-white/20'}`}
                              >
                                <WaveformSymbol type={type} active={settings.carrierType === type} />
                                <span className="text-[6px] font-black uppercase tracking-tighter">{type}</span>
                              </button>
                           ))}
                        </div>
                      </div>
                    </div>
                 </ControlGroup>
              </div>

              <div className="rack-module p-5 border-l-4 border-indigo-500 bg-black/40 relative">
                 <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                       <Combine size={20} className="text-indigo-500" />
                       <div>
                          <h3 className="text-[10px] font-black font-orbitron tracking-[0.2em] text-slate-400 uppercase">Harmonizer Matrix</h3>
                       </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <button
                        onClick={() => updateSetting('harmonizerExtreme', !settings.harmonizerExtreme)}
                        className={`flex items-center gap-2 px-3 py-1 rounded text-[8px] font-black uppercase transition-all ${settings.harmonizerExtreme ? 'bg-orange-500 text-black' : 'bg-white/5 text-slate-500'}`}
                      >
                         <Flame size={10} /> Extreme
                      </button>
                      <div
                        onClick={() => updateSetting('harmonizerEnabled', !settings.harmonizerEnabled)}
                        className={`w-12 h-6 rounded-full p-1 cursor-pointer transition-all ${settings.harmonizerEnabled ? 'bg-indigo-500' : 'bg-slate-800'}`}
                      >
                        <div className={`w-4 h-4 bg-white rounded-full transition-all ${settings.harmonizerEnabled ? 'translate-x-6' : 'translate-x-0'}`} />
                      </div>
                    </div>
                 </div>
                 <div className="grid grid-cols-4 gap-4">
                    {settings.harmonizerVoices.map((voice, idx) => (
                      <div key={idx} className={`p-3 rounded border transition-all ${voice.enabled ? 'border-indigo-500/30 bg-indigo-500/5' : 'border-white/5 bg-black/20 opacity-50'}`}>
                         <div className="flex flex-col gap-3 items-center">
                            <Knob
                              label={`V${idx + 1}`}
                              value={voice.frequency}
                              min={0}
                              max={3000}
                              step={1}
                              unit="Hz"
                              color="#6366f1"
                              onChange={(v) => updateHarmonizerVoice(idx, { frequency: v, enabled: true })}
                            />
                            <div className="w-full flex flex-col gap-2">
                               <Control label="Vol" value={voice.gain} min={0} max={1} step={0.01} color="#6366f1" onChange={(v) => updateHarmonizerVoice(idx, { gain: v })} />
                               <Control label="Pan" value={voice.pan} min={-1} max={1} step={0.01} color="#6366f1" onChange={(v) => updateHarmonizerVoice(idx, { pan: v })} />
                            </div>
                         </div>
                      </div>
                    ))}
                 </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                 <ControlGroup title="Granular Engine" accentClass="border-orange-500">
                    <div className="flex flex-wrap gap-6">
                       <Knob label="Size" value={settings.grainSize} min={0.01} max={0.4} step={0.01} unit="s" color="#ff3e00" onChange={(v) => updateSetting('grainSize', v)} />
                       <Knob label="Density" value={settings.grainDensity} min={5} max={100} step={1} unit="Hz" color="#ff3e00" onChange={(v) => updateSetting('grainDensity', v)} />
                       <Knob label="Pitch" value={settings.grainPitch} min={0.5} max={4.0} step={0.01} unit="x" color="#ff3e00" onChange={(v) => updateSetting('grainPitch', v)} />
                    </div>
                 </ControlGroup>

                 <div className="rack-module p-5 border-l-4 border-rose-500 bg-black/40 relative">
                    <div className="flex items-center justify-between mb-4">
                       <div className="flex items-center gap-3">
                          <Disc size={20} className={`text-rose-500 ${isRecordingLoop ? 'animate-spin' : ''}`} />
                          <h3 className="text-[10px] font-black font-orbitron tracking-[0.2em] text-slate-400 uppercase">Looper</h3>
                       </div>
                       <button onClick={toggleRecording} className={`px-3 py-2 rounded text-[9px] font-black uppercase transition-all ${isRecordingLoop ? 'bg-rose-600 text-white animate-pulse' : 'bg-rose-500 text-black hover:bg-white'}`}>
                         {isRecordingLoop ? 'STOP' : 'REC'}
                       </button>
                    </div>
                    <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
                       {loops.map((loop, idx) => (
                          <div key={loop.id} className="flex-shrink-0 flex gap-2 items-center bg-black/60 p-2 rounded border border-white/5">
                             <span className="text-[9px] text-rose-500 font-black">L{idx + 1}</span>
                             <button onClick={() => toggleMute(loop.id)} className={`p-1 rounded ${loop.isMuted ? 'text-rose-900' : 'text-slate-400'}`}>
                                {loop.isMuted ? <VolumeX size={10} /> : <Volume2 size={10} />}
                             </button>
                             <button onClick={() => removeLoop(loop.id)} className="text-slate-600 hover:text-red-500"><Trash2 size={10} /></button>
                          </div>
                       ))}
                    </div>
                 </div>
              </div>

              {/* AGENTIC BASIN MODULE INTEGRATION */}
              <BasinModule
                basin={basinState}
                fmj={fmjState}
                crispr={crisprState}
                director={directorState}
                neuralSuit={neuralSuitState}
                onToggleModule={toggleBasinModule}
              />
           </div>

           <div className="lg:col-span-4 flex flex-col gap-4">
              <div className="rack-module p-4 flex flex-col gap-6 border-l-2 border-lime-400">
                 <h3 className="text-[9px] font-black font-orbitron tracking-[0.3em] text-slate-500 uppercase border-b border-white/5 pb-2">Master</h3>
                 <div className="flex flex-col gap-4">
                    <Control label="Dry/Wet" value={settings.mix} min={0} max={1} step={0.01} color="#ccff00" onChange={(v) => updateSetting('mix', v)} />
                    <Control label="Output" value={settings.outputGain} min={0} max={2.0} step={0.01} color="#ccff00" onChange={(v) => updateSetting('outputGain', v)} />
                 </div>
              </div>

              {/* 3D SYNTHESIS MODULE */}
              <div className="rack-module p-5 border-l-4 border-blue-500 bg-black/40 relative">
                 <div className="flex items-center gap-3 mb-4">
                    <Maximize size={16} className="text-blue-500" />
                    <h3 className="text-[10px] font-black font-orbitron tracking-[0.2em] text-slate-400 uppercase">3D Synthesis</h3>
                 </div>
                 <div className="flex justify-around items-start gap-4">
                    <Knob
                      label="Width"
                      value={settings.dimX}
                      min={0}
                      max={1}
                      step={0.01}
                      color="#3b82f6"
                      onChange={(v) => updateSetting('dimX', v)}
                    />
                    <Knob
                      label="Depth"
                      value={settings.dimY}
                      min={0}
                      max={1}
                      step={0.01}
                      color="#3b82f6"
                      onChange={(v) => updateSetting('dimY', v)}
                    />
                    <Knob
                      label="Dimension"
                      value={settings.dimZ}
                      min={0}
                      max={1}
                      step={0.01}
                      color="#3b82f6"
                      onChange={(v) => updateSetting('dimZ', v)}
                    />
                 </div>
                 <div className="flex justify-between px-2 pt-2">
                    <MoveHorizontal size={10} className="text-slate-700" />
                    <Box size={10} className="text-slate-700" />
                    <Layers size={10} className="text-slate-700" />
                 </div>
              </div>

              <div className="rack-module p-5 border-l-4 border-[#ffaa00] bg-black/40">
                 <div className="flex items-center justify-between mb-4">
                    <h3 className="text-[10px] font-black font-orbitron text-slate-400 uppercase">Chain Lab</h3>
                    <div className="flex gap-1">
                      <button onClick={captureNode} className="p-2 bg-white/5 border border-white/10 text-slate-300 rounded hover:bg-white hover:text-black transition-all"><Plus size={10} /></button>
                      <button onClick={isPlayingChain ? stopChain : triggerChain} className={`p-2 rounded transition-all ${isPlayingChain ? 'bg-red-500' : 'bg-[#ffaa00]'}`}>
                        {isPlayingChain ? <SquareIcon size={10} /> : <Play size={10} />}
                      </button>
                    </div>
                 </div>
                 <div className="flex gap-1 overflow-x-auto pb-2">
                    {chain.map((step, idx) => (
                      <div key={step.id} className={`flex-shrink-0 w-6 h-6 rounded border flex items-center justify-center text-[7px] font-black ${idx === activeStepIndex ? 'bg-[#ffaa00] text-black' : 'bg-black border-white/10 text-slate-600'}`}>
                        {idx + 1}
                      </div>
                    ))}
                 </div>
              </div>

              <div className="rack-module p-4 border-l-2 border-[#bf00ff] bg-[#bf00ff]/5">
                 <h3 className="text-[9px] font-black font-orbitron text-slate-400 uppercase mb-4">Neural Engine</h3>
                 <div className="flex flex-col gap-2">
                    <input
                      type="text"
                      value={personaPrompt}
                      onChange={(e) => setPersonaPrompt(e.target.value)}
                      placeholder="Prompt..."
                      className="bg-black/50 border border-white/10 rounded px-3 py-2 text-[10px] font-mono text-[#bf00ff] focus:outline-none"
                    />
                    <button onClick={handlePersonaGen} disabled={isPersonaLoading} className="bg-[#bf00ff] text-black py-2 rounded text-[9px] font-black uppercase flex items-center justify-center gap-2">
                       {isPersonaLoading ? <RefreshCw className="animate-spin" size={12} /> : <Zap size={12} />} Synthesize
                    </button>
                 </div>
              </div>

              <div className="rack-module p-4 flex flex-col flex-1 overflow-hidden">
                 <div className="flex items-center justify-between mb-4">
                   <h3 className="text-[9px] font-black uppercase tracking-[0.3em] text-slate-500">Presets</h3>
                   <div className="flex gap-1">
                      <input
                        type="text"
                        placeholder="Name..."
                        value={newPresetName}
                        onChange={(e) => setNewPresetName(e.target.value)}
                        className="bg-black/40 border border-white/5 rounded px-2 py-1 text-[8px] font-mono text-white/80 focus:outline-none focus:border-cyan-500/50 w-24"
                      />
                      <button
                        onClick={savePreset}
                        className="p-1 text-cyan-500 hover:text-white transition-colors"
                        title="Save Current Settings"
                      >
                        <Save size={12} />
                      </button>
                   </div>
                 </div>

                 <div className="flex flex-col gap-1 overflow-y-auto custom-scrollbar pr-2 max-h-[250px]">
                    <div className="text-[7px] font-bold text-slate-600 uppercase tracking-widest mb-1 pl-1">Factory</div>
                    {PRESETS.map(p => (
                      <button
                        key={p.name}
                        onClick={() => { setSettings(p.settings); engineRef.current?.updateSettings(p.settings); }}
                        className={`p-2 text-left rounded border transition-all text-[9px] font-black uppercase tracking-tighter ${settings === p.settings ? 'border-[#00f2ff]/40 bg-[#00f2ff]/5 text-[#00f2ff]' : 'border-white/5 text-slate-500 hover:border-white/10 hover:text-slate-300'}`}
                      >
                         {p.name}
                      </button>
                    ))}

                    {userPresets.length > 0 && (
                      <>
                        <div className="text-[7px] font-bold text-slate-600 uppercase tracking-widest mt-4 mb-1 pl-1">User</div>
                        {userPresets.map(p => (
                          <div key={p.name} className="flex gap-1 group">
                            <button
                              onClick={() => { setSettings(p.settings); engineRef.current?.updateSettings(p.settings); }}
                              className={`flex-1 p-2 text-left rounded border transition-all text-[9px] font-black uppercase tracking-tighter ${settings === p.settings ? 'border-[#00f2ff]/40 bg-[#00f2ff]/5 text-[#00f2ff]' : 'border-white/5 text-slate-500 hover:border-white/10 hover:text-slate-300'}`}
                            >
                               {p.name}
                            </button>
                            <button
                              onClick={() => deleteUserPreset(p.name)}
                              className="p-2 text-slate-700 hover:text-red-500 transition-colors"
                            >
                              <X size={10} />
                            </button>
                          </div>
                        ))}
                      </>
                    )}
                 </div>
              </div>
           </div>
        </div>

        <footer className="bg-black p-3 flex justify-between items-center font-mono text-[8px] text-slate-600 uppercase border-t border-white/5">
           <div className="flex gap-8 pl-4">
              <span className="flex items-center gap-2"><Cpu size={12} /> STABLE</span>
              <span className="flex items-center gap-2"><Tally4 size={12} /> POLY: 4</span>
           </div>
           <div className="pr-4 flex items-center gap-4 italic opacity-50">
              VOXGRAIN_X // v2.6.0
              <div className="w-2 h-2 rounded-full bg-green-500 pulse-led" />
           </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
