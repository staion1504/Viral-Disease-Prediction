import { Routes, Route } from "react-router-dom";
import Predictor_Home from "./pages/Predictor_Home";

function App() {
  return (
    <div>
      <Routes>
          <Route path="/" element={<Predictor_Home />} />
        </Routes>
    </div>
  );
}

export default App;
