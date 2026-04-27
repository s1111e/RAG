import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
import time

endpoints = [
    {
        "name": "1. Llama 3.1 8B",
        "url": "http://149.165.173.247:8888/v1",
        "key": "utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem",
        "model": "meta-llama/Llama-3.1-8B-Instruct"
    },
    {
        "name": "2. Qwen3-8B",
        "url": "http://149.165.171.140:8888/v1",
        "key": "utsa-08GdYYyq2lzmWc02fhfMSKzv3ACPwYgq6U02BozaaupZym1wGQzJBNC59dV4wFTi",
        "model": "Qwen/Qwen3-8B"
    },
    {
        "name": "3. OpenAI GPT OSS 20B (VPN)",
        "url": "http://10.100.1.212:8888/v1",
        "key": "utsa-08GdYYyq2lzmWc02fhfMSKzv3ACPwYgq6U02BozaaupZym1wGQzJBNC59dV4wFTi",
        "model": "openai/gpt-oss-20b"
    },
    {
        "name": "4. Llama 3.3 70B (VPN)",
        "url": "http://10.246.100.230/v1",
        "key": "gpustack_50e00c9281422bc5_0c0696dfcb1696d7635e58a2e56d6282",
        "model": "llama-3.3-70b-instruct-awq"
    },
    {
        "name": "5. Qwen3.5-27B (VPN)",
        "url": "http://10.100.1.213:8888/v1",
        "key": "utsa-08GdYYyq2lzmWc02fhfMSKzv3ACPwYgq6U02BozaaupZym1wGQzJBNC59dV4wFTi",
        "model": "Qwen/Qwen3.5-27B"
    }
]

def test_endpoint(endpoint):
    print(f"\n{'='*60}")
    print(f"Testing: {endpoint['name']}")
    print(f"URL: {endpoint['url']}")
    print(f"Model: {endpoint['model']}")
    print('='*60)
    
    try:
        client = OpenAI(
            base_url=endpoint['url'],
            api_key=endpoint['key'],
            timeout=10.0
        )
        
        response = client.chat.completions.create(
            model=endpoint['model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning? Answer in one sentence."}
            ],
            temperature=0.2,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content
        print(f"✅ SUCCESS!\n")
        print(f"Response: {answer}\n")
        return True, endpoint
        
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "ConnectTimeout" in error_msg or "Name or service not known" in error_msg:
            print(f"❌ CONNECTION FAILED: Endpoint unreachable\n")
        else:
            print(f"❌ ERROR: {error_msg}\n")
        return False, None

def main():
    print("\n" + "="*60)
    print("TESTING ALL LLM ENDPOINTS FROM keys.txt")
    print("="*60)
    
    working_endpoint = None
    
    for endpoint in endpoints:
        success, ep = test_endpoint(endpoint)
        if success:
            working_endpoint = ep
            break
    
    print("\n" + "="*60)
    if working_endpoint:
        print("✅ WORKING ENDPOINT FOUND!")
        print(f"\nUpdate config.py with:")
        print(f"LLM_API_URL = \"{working_endpoint['url']}\"")
        print(f"LLM_API_KEY = \"{working_endpoint['key']}\"")
        print(f"LLM_MODEL = \"{working_endpoint['model']}\"")
    else:
        print("❌ NO WORKING ENDPOINTS FOUND")
        print("\nAll endpoints are either unavailable or require VPN connection.")
        print("Please ensure you are connected to UTSA VPN.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
