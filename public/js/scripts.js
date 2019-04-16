$(document).ready(function(){

    var sentences = ["Saab", "Volvo", "BMW"];
    var n = sentences.length;
    var rand;

    $( "#shuffle" ).click(function() {
      rand = Math.floor(Math.random() * n);
      $('#input').val(sentences[rand]);
    });

});
