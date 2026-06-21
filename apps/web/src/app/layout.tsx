import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AURA | Research paper Q&A",
  description: "Paper-grounded answers across the AURA research corpus.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
