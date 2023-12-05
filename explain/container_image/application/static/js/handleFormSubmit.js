/* https://simonplend.com/how-to-use-fetch-to-post-form-data-as-json-to-your-api/
/**
 * Helper function for POSTing data as JSON with fetch.
 *
 * @param {Object} options
 * @param {string} options.url - URL to POST data to
 * @param {FormData} options.formData - `FormData` instance
 * @return {Object} - Response body from URL that was POSTed to
 */
async function sendFormDataAsJson(url, method, formData) {
  const plainFormData = Object.fromEntries(formData.entries());
  const formDataJsonString = JSON.stringify(plainFormData);

  const fetchOptions = {
    method: method,
    headers: {
      "Content-Type": "application/json",
      Accept: "text/html",
    },
    redirect: "error",
    body: formDataJsonString,
  };
  console.log("payload: ");
  console.log(fetchOptions);
  const response = await fetch(url, fetchOptions);

  console.log(response);

  if (!response.ok) {
    const errorMessage = await response.text();
    alert("Error happend on the submit!");
    throw new Error(errorMessage);
  } else {
    window.location.replace(response.headers.get("Location"));
  }
}
