const form = document.getElementById("sentiment-form");
const input = document.getElementById("input-text");
const output = document.getElementById("output-text");

form.addEventListener("submit", async function(e){


  e.preventDefault();

  output.value = "Predicting Review...";
  
  const formdata = new FormData(form);
  const response = await fetch("/upload",{
       method: "POST",
       body : formdata
  });

  const data = await response.text();
  output.value = data;

});