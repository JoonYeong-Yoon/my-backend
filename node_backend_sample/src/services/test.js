const testModel = require("../models/test");
async function getAllDBLists() {
  const db_lists = await testModel.getAllDB();
  // 쿼리 호출 후에 추가 작업 필요하면 추가작업
  return db_lists;
}

module.exports = { getAllDBLists };
