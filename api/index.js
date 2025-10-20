// This is a minimal serverless function to handle all API routes
module.exports = (req, res) => {
  res.status(200).json({ message: "API is working" });
}
