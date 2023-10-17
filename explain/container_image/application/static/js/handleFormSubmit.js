/* https://simonplend.com/how-to-use-fetch-to-post-form-data-as-json-to-your-api/
/**
 * Helper function for POSTing data as JSON with fetch.
 *
 * @param {Object} options
 * @param {string} options.url - URL to POST data to
 * @param {FormData} options.formData - `FormData` instance
 * @return {Object} - Response body from URL that was POSTed to
 */
async function postFormDataAsJson(url, method, formData) {
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

/**
 * Event handler for a form submit event.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/HTMLFormElement/submit_event
 *
 * @param {SubmitEvent} event
 */
async function handleFormSubmit(event) {
  console.log(event);
  event.preventDefault();
  event.submitter.disabled = true;

  const form = event.target;
  const url = form.action;

  try {
    const formData = new FormData(form);
    await postFormDataAsJson({ url, formData });
    return false;
  } catch (error) {
    console.log(error);
  }
}
