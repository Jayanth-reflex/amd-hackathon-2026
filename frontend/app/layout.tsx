import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Qwen3.6-35B-A3B-Domain-Aggressive — AMD Hackathon 2026 demo",
  description:
    "Domain-specialized, abliterated MoE on AMD MI300X. Capability in the model, policy at the gateway.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-zinc-950 text-zinc-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
