import { ReactComponent as Microphone } from "./assets/Microphone.svg";
import { ReactMediaRecorder } from "react-media-recorder";
import { useState } from "react";

type Props = {
  handleStop: any;
};

function Recorder({ handleStop }: Props) {
  const [isRecording, setIsRecording] = useState(false);

  const handleClick = (
    startRecording: { (): void; (): void },
    stopRecording: { (): void; (): void }
  ) => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
    setIsRecording(!isRecording);
  };
  return (
    <ReactMediaRecorder
      audio
      onStop={handleStop}
      render={({ status, startRecording, stopRecording }) => (
        <div className="mt-2">
          <button
            onClick={() => handleClick(startRecording, stopRecording)}
            className="bg-white p-4 rounded-full text-teal-300"
          >
            <Microphone
              className={
                status == "recording"
                  ? "animate-pulse text-red-500 hover:text-blue-950"
                  : "text-teal-300 hover:text-blue-950"
              }
            />
          </button>
        </div>
      )}
    />
  );
}

export default Recorder;
