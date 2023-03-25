import React, { useState } from "react";
import axios from "axios";
import { Button, Container, Grid, Paper, Typography } from "@mui/material";
import "./App.css";

function App() {
  const [capturedImage, setCapturedImage] = useState(null);
  const [returnedImage, setReturnedImage] = useState(null);
  const [landscapeInfo, setLandscapeInfo] = useState(null);

  const handleSubmit = async (file) => {
    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await axios.post("/api", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { class: className, cam: camBase64 } = response.data;

      setReturnedImage(`data:image/jpeg;base64,${camBase64}`);
      setLandscapeInfo(className);
    } catch (error) {
      console.error("Error during image classification:", error);
    }
  };

  const handleImageChange = async (event) => {
    const file = event.target.files[0];
    setCapturedImage(file);
    setReturnedImage(null);
    setLandscapeInfo(null);
    await handleSubmit(file);
  };

  return (
    <Container maxWidth="md" className="container">
      <Typography variant="h4" component="h1" gutterBottom>
        PolyU Landscapes
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <Paper elevation={2} className="paper">
            <Typography variant="h6" component="h2" gutterBottom>
              Landscape
            </Typography>
            {capturedImage && (
              <img
                className="image"
                src={URL.createObjectURL(capturedImage)}
                alt="Captured"
              />
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Paper elevation={2} className="paper">
            <Typography variant="h6" component="h2" gutterBottom>
              Heat Map
            </Typography>
            {returnedImage && (
              <img className="image" src={returnedImage} alt="Returned" />
            )}
          </Paper>
        </Grid>
      </Grid>
      <div className="actions">
        <input
          id="capture-image-input"
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleImageChange}
          hidden
        />
        <label htmlFor="capture-image-input">
          <Button variant="contained" color="primary" component="span">
            Choose Image
          </Button>
        </label>
      </div>
      <div className="landscape-info-container">
        <Typography variant="h6" component="h2" gutterBottom>
          Landscape Information
        </Typography>
        {landscapeInfo && (
          <div className="landscape-info">
            <Typography>{landscapeInfo}</Typography>
          </div>
        )}
      </div>
    </Container>
  );
}

export default App;
