"""Show which models this API key can reach, and generate one test scene."""
import sys
from forge.synth.teacher import Teacher

provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
models = Teacher.list_models(provider)
print(f"{len(models)} models reachable with your {provider} key:\n")
for m in models:
    print("  " + m)
