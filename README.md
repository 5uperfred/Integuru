# Integuru

An AI agent that generates integration code by reverse-engineering platforms' internal APIs.

## Integuru in Action

![Integuru in action](./integuru_demo.gif)

## What Integuru Does

You use `create_har.py` to generate a file containing all browser network requests, a file with the cookies, and write a prompt describing the action triggered in the browser. The agent outputs runnable Python code that hits the platform's internal endpoints to perform the desired action.

## How It Works

Let's assume we want to download utility bills:

1.  The agent identifies the request that downloads the utility bills. For example, the request URL might look like this:
    `https://www.example.com/utility-bills?accountId=123&userId=456`

2.  It identifies parts of the request that depend on other requests. The above URL contains dynamic parts (accountId and userId) that need to be obtained from other requests.
    `accountId=123`
    `userId=456`

3.  It finds the requests that provide these parts and makes the download request dependent on them. It also attaches these requests to the original request to build out a dependency graph.
    `GET https://www.example.com/get_account_id`
    `GET https://www.example.com/get_user_id`

4.  This process repeats until the request being checked depends on no other request and only requires the authentication cookies.

5.  The agent traverses up the graph, starting from nodes (requests) with no outgoing edges until it reaches the master node while converting each node to a runnable function.

## Features

-   Generate a dependency graph of requests to make the final request that performs the desired action.
-   Allow input variables (for example, choosing the YEAR to download a document from). This is currently only supported for graph generation. Input variables for code generation coming soon!
-   Generate code to hit all requests in the graph to perform the desired action.

## Setup

1.  Copy the `.env.example` file to a new file named `.env` and add your API key.
    ```bash
    cp .env.example .env
    ```
    You will need to add your `GOOGLE_API_KEY` to this file.

2.  Install Python requirements via poetry:
    ```
    poetry install
    ```

3.  Open a poetry shell:
    ```
    poetry shell
    ```

4.  Register the Poetry virtual environment with Jupyter:
    ```
    poetry run ipython kernel install --user --name=integuru
    ```

5.  Run the following command to spawn a browser:
    ```
    poetry run python create_har.py
    ```
    Log into your platform and perform the desired action (such as downloading a utility bill).

6.  Run Integuru:
    ```
    poetry run integuru --prompt "download utility bills"
    ```
    The agent defaults to `gemini-2.5-flash`. If you want to use a different model, you can specify it:
    ```
    poetry run integuru --prompt "download utility bills" --model gemini-2.5-pro
    ```
    You can also run it via Jupyter Notebook `main.ipynb`

> **Note on Models**
>
> Integuru is optimized for Google's Gemini models and uses the "thinking" feature for improved reasoning.
>
> * **`gemini-2.5-flash`** (Default): Used for all standard graph generation and reasoning steps.
> * **`gemini-2.5-pro`**: Automatically used for the final code generation step due to its stronger coding capabilities (as defined in `LLM.py`).
