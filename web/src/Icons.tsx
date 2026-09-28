// Inline SVG icons (no icon font or CDN: the workbench runs offline, DESIGN.md §2.8). 24x24 stroke icons.
type P = { size?: number; className?: string };
const I = ({ size = 16, className, d }: P & { d: string }) => (
  <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
);
export const Search = (p: P) => <I {...p} d="M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm10 2-4.35-4.35" />;
export const Layers = (p: P) => <I {...p} d="m12 2 10 5-10 5L2 7l10-5Zm-10 10 10 5 10-5M2 17l10 5 10-5" />;
export const Similar = (p: P) => <I {...p} d="M3 3h7v7H3zM14 14h7v7h-7zM14 3h7v7h-7zM3 14h7v7H3z" />;
export const Group = (p: P) => <I {...p} d="M8 12a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm8 6a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM6 20a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm12-12a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />;
export const Check = (p: P) => <I {...p} d="M20 6 9 17l-5-5" />;
export const X = (p: P) => <I {...p} d="M18 6 6 18M6 6l12 12" />;
export const Download = (p: P) => <I {...p} d="M12 3v12m0 0-4-4m4 4 4-4M4 21h16" />;
export const Gear = (p: P) => <I {...p} d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.3 7.3 0 0 0-2-1.2L14.5 3h-5l-.4 2.6a7.3 7.3 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7.3 7.3 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7.3 7.3 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2Z" />;
export const Help = (p: P) => <I {...p} d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Zm-3-13a3 3 0 1 1 4.2 2.8c-.7.3-1.2 1-1.2 1.7V14m0 3h.01" />;
export const Filter = (p: P) => <I {...p} d="M3 5h18l-7 8v6l-4 2v-8L3 5Z" />;
export const Back = (p: P) => <I {...p} d="M19 12H5m6-6-6 6 6 6" />;
export const Change = (p: P) => <I {...p} d="M3 12h4l3-8 4 16 3-8h4" />;
export const Chart = (p: P) => <I {...p} d="M3 3v18h18M7 15l4-4 3 3 5-6" />;
export const Shield = (p: P) => <I {...p} d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />;
export const Arrow = (p: P) => <I {...p} d="M5 12h14m-6-6 6 6-6 6" />;
export const Logo = ({ size = 22 }: P) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <circle cx="16" cy="16" r="14" fill="none" stroke="#f5b400" strokeWidth="2.5" />
    <path d="M6 20c4-3 7-3 10 0s6 3 10 0" fill="none" stroke="#58a6ff" strokeWidth="2.5" strokeLinecap="round" />
    <rect x="11" y="8" width="10" height="7" rx="1.5" fill="#f5b400" />
  </svg>
);
