/*
npm init
npm install --save express
npm install request --save
npm install ejs --save
npm install body-parser --save
*/

const express = require('express');
const bodyParser = require('body-parser');
const request = require('request');
const spawn = require('child_process').spawn;
const app = express()

// Give Express access to public folder
app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));
// Set up a template engine
app.set('view engine', 'ejs')

// Render templates for get and post requests
app.get('/', function (req, res) {
  res.render('index', {segments: null});
})

app.post('/', function (req, res) {
  let sentence = req.body.sentence;

  // Run split.py
  let py    = spawn('python', ['split.py']);
  let dataString = '';
  let arr = null;

  // Receive data
  py.stdout.on('data', function(data){
    dataString += data.toString();
  });
  py.stdout.on('end', function(){
      // Split the dataString's lines and slice off the last empty element
      arr = dataString.split(/\r?\n/).slice(0, -1);

      res.render('index', {segments: arr});
  });
  py.stdin.write(JSON.stringify(sentence));
  py.stdin.end();

})

app.listen(3000, function () {
  console.log('Example app listening on port 3000!')
})
