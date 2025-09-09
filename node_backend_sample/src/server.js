const express = require("express");
const { corsMiddleware, cookieMiddleware } = require("./middlewares");
const testRoutes = require("./routes/test");
const app = express();
require("dotenv").config();

app.use(corsMiddleware);
app.use(cookieMiddleware);
app.use(express.json({ limit: "50mb" }));
app.use("/", testRoutes);
// app.use("/user", userRoutes);

const PORT = require("./config").BACKEND_PORT;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
