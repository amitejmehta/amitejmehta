// src/components/Section1.tsx
import React from "react";
import Hexagon from "./Hexagon";
import Suitcase from "./Suitcase";

const Section1: React.FC = () => {
  return (
    <div
      className="flex justify-center items-center bg-cover bg-top h-[100vh] md:flex-row flex-col"
      style={{ backgroundImage: 'url("/night.png")' }}
    >
      <Hexagon
        sectionId="section-good"
        icon={<Suitcase className="w-16 h-16" />}
      />
      <Hexagon
        sectionId="section-business"
        icon={<Suitcase className="w-16 h-16" />}
      />
      <Hexagon
        sectionId="section-unknown"
        icon={<Suitcase className="w-16 h-16" />}
      />
    </div>
  );
};

export default Section1;
