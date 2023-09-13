import * as React from "react";
import { SVGProps } from "react";
const Suitcase = (props: SVGProps<SVGSVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={200}
    height={200}
    viewBox="0 0 32 32"
    {...props}
  >
    <title>{"suitcase1"}</title>
    <path d="M27 29H4a2 2 0 0 1-2-2V15s5.221 2.685 10 3.784V20a1 1 0 0 0 1 1h5a1 1 0 0 0 1-1v-1.216C23.778 17.685 29 15 29 15v12a2 2 0 0 1-2 2zM17 17a1 1 0 0 1 1 1v1a1 1 0 0 1-1 1h-3a1 1 0 0 1-1-1v-1a1 1 0 0 1 1-1h3zm2 0a1 1 0 0 0-1-1h-5a1 1 0 0 0-1 1v.896C7.221 16.764 2 14 2 14v-4a2 2 0 0 1 2-2h6V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v2h6a2 2 0 0 1 2 2v4s-5.222 2.764-10 3.896V17zm0-10a1 1 0 0 0-1-1h-5a1 1 0 0 0-1 1v1h7V7z" />
  </svg>
);
export default Suitcase;
