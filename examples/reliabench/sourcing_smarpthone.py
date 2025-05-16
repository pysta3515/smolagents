from smolagents import Tool


countries = ["France", "Germany", "Italy", "Spain", "United Kingdom", "United States"]

countries = ['USA', 'Japan', 'Germany', 'India']
final_prices = {}

for country in countries:
   exchange_rate, tax_rate = lookup_rates(country)
   local_price = lookup_phone_price("XAct 1", country)
   converted_price = convert_and_tax(
       local_price, exchange_rate, tax_rate
   )
   shipping_cost = estimate_shipping_cost(country)
   final_price = estimate_final_price(converted_price, shipping_cost)
   final_prices[country] = final_price

most_cost_effective_country = min(final_prices, key=final_prices.get)
most_cost_effective_price = final_prices[most_cost_effective_country]
print(most_cost_effective_country, most_cost_effective_price)

if __name__ == "__main__":
    task = 
