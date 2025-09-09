const pool = require("../db/pool");
async function getAllDB() {
  const rows = await pool.execute("SHOW databases;");
  return rows;
}

module.exports = { getAllDB };
