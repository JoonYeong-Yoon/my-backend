const cors = require("cors");
const config = require("../config");
const cookieParser = require("cookie-parser");

const corsOptions = {
  origin: [`http://localhost:${config.FRONT_PORT}`, "https://blog.cjcho.site"],
  credentials: true,
};

module.exports = {
  corsMiddleware: cors(corsOptions),
  cookieMiddleware: cookieParser(),
};
