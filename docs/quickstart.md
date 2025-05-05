## Quick Start

This section guides you through installing KnowOrNot and setting up your first LLM client connection, using Azure OpenAI as an example.

### 1. Installation

Install KnowOrNot using pip:

```bash
pip install knowornot
```

Install KnowOrNot using uv
```bash
uv add knowornot
```

*(Note: If you are developing locally and have cloned the repository, you might use `uv add ../KnoworNot` or `pip install -e ../KnoworNot` from your project's root directory)*

### 2. Initialize the KnowOrNot Instance

Import the main class and create an instance.

```python
from knowornot import KnowOrNot

# Initialize KnowOrNot
kon = KnowOrNot()

print("KnowOrNot instance initialized.")
```

### 3. Add an Azure OpenAI Client

KnowOrNot requires a connection to an LLM to perform operations like generating questions or running experiments. The `add_azure` method is a convenient way to configure and register an Azure OpenAI client. This method automatically registers the client within the `KnowOrNot` instance and sets it as the default client unless specified otherwise.

You can provide your Azure credentials and configuration in two ways:

#### Option A: Using Environment Variables (Recommended)

Store your sensitive credentials and configuration in a `.env` file. This is generally the most secure approach.

**Example `.env` file:**

Create a file named `.env` in the root of your project directory and add your Azure details:

```dotenv
AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE_NAME.openai.azure.com/"
AZURE_OPENAI_API_KEY="YOUR_API_KEY"
AZURE_OPENAI_API_VERSION="YYYY-MM-DD" # e.g., "2024-02-15" - Use a recent API version

AZURE_OPENAI_DEFAULT_MODEL="YOUR_CHAT_DEPLOYMENT_NAME"
AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL="YOUR_EMBEDDING_DEPLOYMENT_NAME"
```

Replace the placeholder values with your actual Azure OpenAI resource information and deployment names.

In your Python script, you'll need to load these environment variables (e.g., using the `python-dotenv` library: `pip install python-dotenv`). Then, call `kon.add_azure()` without any arguments. It will automatically read the configuration from the environment variables.

```python
import os
from knowornot import KnowOrNot
from dotenv import load_dotenv # Requires `pip install python-dotenv`

# Load environment variables from a .env file (if using this method)
load_dotenv()

# Initialize KnowOrNot
kon = KnowOrNot()

# Add Azure client using environment variables
# The add_azure method will look for AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, etc.
try:
    kon.add_azure()
    print("Azure client added using environment variables.")

    # You can get the default client to confirm
    default_client = kon.get_client()
    print(f"Default client registered: {default_client.enum_name}")

except EnvironmentError as e:
    print(f"Error adding Azure client: {e}")
    print("Please ensure required environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION) are set or provided as parameters.")
except ImportError:
     print("Error: python-dotenv not found. Please install it (`pip install python-dotenv`) or use the direct parameter method.")


```

#### Option B: Passing Parameters Directly

Alternatively, you can pass the configuration details directly as arguments to the `add_azure` method. This is less secure for API keys in code but might be useful for testing or specific deployment scenarios where environment variables aren't feasible.

```python
import os
from knowornot import KnowOrNot

# Initialize KnowOrNot
kon = KnowOrNot()

# Add Azure client by passing parameters directly
try:
    kon.add_azure(
        azure_endpoint="https://YOUR_RESOURCE_NAME.openai.azure.com/",
        azure_api_key="YOUR_API_KEY",
        azure_api_version="YYYY-MM-DD",
        default_model="YOUR_CHAT_DEPLOYMENT_NAME",
        default_embedding_model="YOUR_EMBEDDING_DEPLOYMENT_NAME"
    )
    print("Azure client added by passing parameters directly.")

    # You can get the default client to confirm
    default_client = kon.get_client()
    print(f"Default client registered: {default_client.enum_name}")

except Exception as e:
    print(f"Error adding Azure client: {e}")
    print("Please ensure correct parameters are provided.")

```

After successfully running either Option A or B, your `KnowOrNot` instance is initialized and configured with an Azure OpenAI client, ready for subsequent operations.

You can now proceed with generating questions, setting up experiments, and evaluating results as described in other sections of the documentation.
