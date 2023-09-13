import React from "react";
import { useState } from "react";
import axios from "axios";
import { ReactComponent as Refresh } from "./assets/Refresh.svg";

type Props = {
  setMessages: any;
};

function Title({ setMessages }: Props) {
  const [isResetting, setIsResetting] = useState(false);

  const resetConversation = async () => {
    setIsResetting(true);

    await axios
      .get("http://localhost:8000/new_chat")
      .then((response) => {
        if (response.status == 200) {
          setMessages([]);
        } else {
          console.error(
            "There was an error with the GET request to start a new chat"
          );
        }
      })
      .catch((error) => {
        console.error(error.message);
      });
    setIsResetting(false);
  };

  return (
    <div className="flex justify-between items-center w-full p-4 bg-gradient-to-r from-blue-950 to-teal-300 text-blue-950 font-bold shadow ">
      <div className="flex items-center">
        <div className="-ml-9 mb-2.5 font-bold text-2xl bg-gradient-to-r from-sky-200 via-teal-300 to-sky-600 text-transparent bg-clip-text animate-gradient">
          GPT
        </div>
      </div>
      <button
        onClick={resetConversation}
        className={
          "transition-all duration-300  text-blue-950 hover:text-white " +
          (isResetting && "animate-pulse")
        }
      >
        <Refresh />
      </button>
    </div>
  );
}

export default Title;
