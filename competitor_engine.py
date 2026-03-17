import asyncio
from scraper import scrape_website
from phase2_models import CompareRequest, CompareResult

async def run_competitor_analysis(request: CompareRequest) -> CompareResult:
    print(f"Starting competitor analysis for {request.primary_url}")
    
    # 1. Scrape Primary
    try:
        primary_res = await scrape_website(request.primary_url)
    except Exception as e:
        print(f"Error scraping primary: {e}")
        raise ValueError(f"Failed to scrape primary URL: {str(e)}")
        
    comp_data = []
    
    # 2. Scrape Competitors sequentially to avoid freezing Playwright
    for url in request.competitor_urls:
        url = url.strip()
        if url:
            print(f"Scraping competitor: {url}")
            try:
                res = await scrape_website(url)
                comp_data.append(res)
            except Exception as e:
                print(f"Error scraping competitor {url}: {e}")
                # We can skip failed competitors or append empty
                continue
                
    print(f"Analysis complete. Found {len(comp_data)} competitor results.")
    return CompareResult(
        primary_data=primary_res,
        competitors_data=comp_data
    )
