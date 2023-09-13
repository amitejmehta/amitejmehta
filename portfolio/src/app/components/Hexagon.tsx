// src/components/Hexagon.tsx
"use client";
import React, { ReactNode } from "react";

interface HexagonProps {
  sectionId: string;
  icon?: ReactNode;
  bgColor?: string;
}

const scrollTo = (id: string) => {
  const element = document.getElementById(id);
  if (element) {
    element.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }
};

const Hexagon: React.FC<HexagonProps> = ({
  sectionId,
  icon,
  bgColor = "bg-blue-300",
}) => {
  const handleClick = () => {
    scrollTo(sectionId);
  };

  return (
    <div className="m-4 cursor-pointer text-red-500" onClick={handleClick}>
      <div className="relative w-32 h-18">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-full bg-color-red-500"
          width={100}
          height={100}
          fill="none"
          viewBox="0 0 24 24"
        >
          <path
            stroke="#000"
            className="text-red-500"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M2.461 12.8c-.168-.291-.252-.437-.285-.592a1 1 0 0 1 0-.416c.033-.155.117-.3.285-.592l4.077-7.06c.168-.292.252-.437.37-.543a1 1 0 0 1 .36-.208c.15-.05.319-.05.655-.05h8.153c.336 0 .505 0 .655.05.133.043.256.114.36.208.118.106.202.251.37.543l4.077 7.06c.168.292.252.437.285.592.03.137.03.279 0 .416-.033.155-.117.3-.285.592l-4.076 7.06c-.169.292-.253.438-.37.544a1 1 0 0 1-.36.207c-.151.05-.32.05-.656.05H7.923c-.336 0-.504 0-.655-.05a1 1 0 0 1-.36-.207c-.118-.107-.202-.252-.37-.544L2.46 12.8Z"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          {icon}
        </div>
      </div>
    </div>
  );
};

export default Hexagon;
