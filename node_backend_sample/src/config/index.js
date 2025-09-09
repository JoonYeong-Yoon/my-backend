module.exports = {
  FRONT_PORT: 3000,
  BACKEND_PORT: 5000,
  DB: {
    host: "123.456.789.1",
    port: 1111,
    user: "1111",
    password: "1111",
    database: "111",
    waitForConnections: true,
    connectionLimit: 20,
    queueLimit: 5,
    acquireTimeout: 5000, // 10초 후에 커넥션을 끊도록 설정
    connectTimeout: 5000,
  },

  LOGIN: {
    secret: "111",
    resave: false,
    saveUninitialized: true,
    name: "session-cookie",
    cookie: {
      httpOnly: false,
      secure: false, // HTTPS
      maxAge: 2 * 60 * 10000, //2 시간
    },
  },
};
