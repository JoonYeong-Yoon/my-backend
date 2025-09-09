const testService = require("../services/test");
const jwt = require("jsonwebtoken");

async function test(req, res) {
  try {
    const db_lists = await testService.getAllDBLists();
    res.status(200).json({ message: "Hello World!", data: db_lists });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal Server Error" });
  }
}
module.exports = { test };
