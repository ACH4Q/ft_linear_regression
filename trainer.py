import csv

def scale_values(values):
    min_val = min(values)
    max_val = max(values)
    scaled = [(x - min_val) / (max_val - min_val) for x in values]
    return scaled, min_val, max_val

def descale_theta(theta, min_price, max_price, min_mileage, max_mileage):
    range_price = max_price - min_price
    range_mileage = max_mileage - min_mileage
    theta0_descaled = theta[0] * range_price + min_price - (theta[1] * min_mileage * range_price) / range_mileage
    theta1_descaled = (theta[1] * range_price) / range_mileage
    return theta0_descaled, theta1_descaled


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
scaled_mileage, min_mileage, max_mileage = scale_values(mileage)
scaled_price, min_price, max_price = scale_values(price)
m = len(scaled_mileage)
learning_rate = 0.1
rounds = 1000
theta0 = 0.0
theta1 = 0.0

for i in range(rounds):
    sum_error0 = 0
    sum_error1 = 0
    
    for j in range(m):
        current_mileage = scaled_mileage[j]
        current_price = scaled_price[j]

        prediction = estimed_price(current_mileage, theta0, theta1)
        error = prediction - current_price
        
        sum_error0 += error
        sum_error1 += error * current_mileage

    tmp_theta0 = learning_rate * (1/m) * sum_error0
    tmp_theta1 = learning_rate * (1/m) * sum_error1

    theta0 = theta0 - tmp_theta0 
    theta1 = theta1 - tmp_theta1
    

print("\n -- finished training")
theta0_final, theta1_final = descale_theta([theta0, theta1], min_price, max_price, min_mileage, max_mileage)
print(f"Final learned values: theta0 = {theta0_final:.2f}, theta1 = {theta1_final:.4f}")