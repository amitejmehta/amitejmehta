import { useState } from "react";
import Title from "./Title";
import Recorder from "./Recorder";
import axios from "axios";

function Controller() {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);

  // send recording to backend upon stopping recording
  const createBlobUrl = (data: any) => {
    const blob = new Blob([data], { type: "audio/mpeg" });
    const url = window.URL.createObjectURL(blob);
    return url;
  };

  // send recording to backend upon stopping recording
  const handleStop = async (blobUrl: string) => {
    setIsLoading(true);

    const mymessage = { sender: "me ", blobUrl };
    const messagesArr = [...messages, mymessage];

    fetch(blobUrl)
      .then((response) => response.blob())
      .then(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "myFile.wav");

        await axios
          .post("http://localhost:8000/post-audio", formData, {
            headers: { "Content-Type": "audio/mpeg" },
            responseType: "arraybuffer",
          })
          .then((response: any) => {
            const blob = response.data;
            const audio = new Audio();
            audio.src = createBlobUrl(blob);

            //append to audio
            const aimessage = { sender: "GrooveGPT", blobUrl: audio.src };
            messagesArr.push(aimessage);
            setMessages(messagesArr);

            setIsLoading(false);
            audio.play();
          })
          .catch((error) => {
            console.error(error.message);
            setIsLoading(false);
          });
      });
  };

  return (
    //height of screen; overflow of y axis to be hidden or scroll
    <div className="h-screen overflow-y-hidden">
      <Title setMessages={setMessages} />
      <div className="flex flex-col justify-between h-full overflow-y-scroll pb-96">
        <div className="mt-5 px-5">
          {messages.map((audio, index) => {
            return (
              <div
                key={index + audio.sender}
                className={
                  "flex flex-col " +
                  (audio.sender == "GrooveGPT" ? "items-end" : "")
                }
              >
                <div className="mt-4">
                  <p
                    className={
                      audio.sender == "GrooveGPT"
                        ? "text-right mr-2 italic text-blue-950"
                        : "ml-2 italic text-teal-200"
                    }
                  >
                    {audio.sender}
                  </p>
                  <audio
                    src={audio.blobUrl}
                    className={
                      audio.sender == "GrooveGPT" ? "text-right mr-2" : "ml-2"
                    }
                    controls
                  />
                </div>
              </div>
            );
          })}
          {messages.length == 0 && !isLoading && (
            <div className="text-center font-light mt-10">
              Hey GrooveGPT, can you help me...
            </div>
          )}

          {isLoading && (
            <div className="text-center font-light mt-10 animate-pulse">
              Thinking...how fast can you divide 5289/43?
            </div>
          )}
        </div>
        <div className="fixed bottom-0 w-full py-7 border-t text-center bg-gradient-to-r from-teal-200 to-blue-950">
          <div className="flex justify-center items-center">
            <div>
              <Recorder handleStop={handleStop} />
            </div>
          </div>
        </div>
      </div>
    </div>
    // flex-col = stack vertically justify-between: spreads the divs out; pb: padding bottom
  );
}

export default Controller;
