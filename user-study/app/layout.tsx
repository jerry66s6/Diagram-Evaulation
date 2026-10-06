import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {
  title: 'Diagram Lab · Evaluation Study',
  description: 'A human evaluation study of diagram correctness and visual quality.',
  icons: { icon: '/favicon.svg' },
  robots: {index: false, follow: false},
};
export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body>{children}</body></html>;
}
