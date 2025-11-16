import csv

def load_data(file):
    price = []
    mileage = []

    with open(file, "r") as file_handle:
        reader = csv.reader(file_handle)
        next(reader)
        for row in reader:
            price.append(int(row[1]))
            mileage.append(int(row[0]))
    return price, mileage

def estimed_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

price, mileage = load_data('data.csv') 
m = len(mileage)

learning_rate = 0.0000000001
rounds = 10000
theta0 = 0.0
theta1 = 0.0

for i in range(rounds):
    sum_error0 = 0
    sum_error1 = 0
    
    for j in range(m):
        current_mileage = mileage[j]
        current_price = price[j]

        prediction = estimed_price(current_mileage, theta0, theta1)
        error = prediction - current_price
        
        sum_error0 += error
        sum_error1 += error * current_mileage

    tmp_theta0 = learning_rate * (1/m) * sum_error0
    tmp_theta1 = learning_rate * (1/m) * sum_error1

    theta0 = theta0 - tmp_theta0 
    theta1 = theta1 - tmp_theta1

    print(f"Round {i+1}: theta0 = {theta0:.2f}, theta1 = {theta1:.4f}")

print("\n -- finished training")
print(f"Final learned values: theta0 = {theta0}, theta1 = {theta1}")