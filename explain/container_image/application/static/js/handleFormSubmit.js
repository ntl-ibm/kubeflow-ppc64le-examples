/* https://simonplend.com/how-to-use-fetch-to-post-form-data-as-json-to-your-api/
/**
 * Helper function for POSTing data as JSON with fetch.
 *
 * @param {Object} options
 * @param {string} options.url - URL to POST data to
 * @param {FormData} options.formData - `FormData` instance
 * @return {Object} - Response body from URL that was POSTed to
 */
async function postFormDataAsJson({ url, formData }) {
  const plainFormData = Object.fromEntries(formData.entries());
  const formDataJsonString = JSON.stringify(plainFormData);

  const fetchOptions = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/html",
    },
    redirect: "manual",
    body: formDataJsonString,
  };
  console.log("payload: ");
  console.log(fetchOptions);
  const response = await fetch(url, fetchOptions);

  if (response.headers.has("Location")) {
    replace(response.headers.get("Location"));
  }

  if (!response.ok) {
    const errorMessage = await response.text();
    throw new Error(errorMessage);
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
  console.info(event);
  event.preventDefault();

  const form = event.target;
  const url = form.action;

  try {
    const formData = new FormData(form);
    await postFormDataAsJson({ url, formData });
  } catch (error) {
    console.error(error);
  }
}
