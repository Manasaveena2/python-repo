price=int(input("Enter price:"))
weight=float(input("Enter weight:"))
discount=int(input("Enter discount percentage:"))
tax_rate=float(input("Enter tax_rate:"))
additional_cost=int(input("Enter additional cost:"))
base_shipping_cost=10

total_price=lambda price,discount:price-((discount/100)*price)
print(f"total price after discount is {total_price(price,discount)}")

sales_tax=lambda price,tax_rate:price+(tax_rate*100)
print(f"Total price after adding sales tax is {sales_tax(price,tax_rate)}")

total_shipping_cost=lambda weight,additional_cost,base_shipping_cost:base_shipping_cost+weight*additional_cost
print(f"Total shipping Cost:{total_shipping_cost(weight,additional_cost,base_shipping_cost)}")