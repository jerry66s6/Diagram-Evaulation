import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
  {
    files: ["components/ui/**/*.{ts,tsx}", "hooks/use-mobile.ts"],
    rules: {
      // These files are vendored verbatim from shadcn@4.17.0. Keep the
      // registry source intact while applying the stricter rules to Site code.
      "@typescript-eslint/no-unused-vars": "off",
      "react-hooks/purity": "off",
      "react-hooks/set-state-in-effect": "off",
    },
  },
  {
    rules: {
      // vinext (beta) resolves next/link as a separately optimized dependency in
      // dev, which loads a second React copy and throws "Invalid hook call".
      // Plain anchors do a full navigation between the two pages, which is fine.
      "@next/next/no-html-link-for-pages": "off",
      // Diagrams are local SVG/PNG files shown at native resolution with zoom;
      // next/image optimization adds nothing here.
      "@next/next/no-img-element": "off",
    },
  },
]);

export default eslintConfig;
