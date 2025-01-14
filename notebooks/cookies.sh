#!/bin/bash

# cookies.sh - A script to save credentials as cookies for use of curl to obtain NSIDC-0719 data

# NOTE: Requires an account with NASA EARTHDATA (see https://nsidc.org/home)

# Parameters:
#    user-name (required) - Username to NASA EARTHDATA login
#    password (required) - Password to NASA EARTHDATA login
#    file-name (optional) - Location and file name for where to place cookies. Defaults to $PWD/usr_cookies.txt

# Example usage: bash cookies.sh '<user-name>' '<password>' [file-name]
# NOTE: Keep the quotes around each of the variables if running this in terminal

# Check if the correct number of parameters is provided
if [ $# -lt 2 ]; then
    echo "Requires minimum two inputs for login to NASA Earth Data"
    echo "Usage: bash cookies.sh <user-name> <password> [file (optional)]"
    exit 1
fi

# Access the input parameters
user_name=$1
password=$2
cookies=$3

# Save 

if [ -z "$cookies" ]; then
    cookies=$PWD/usr_cookies.txt
fi

echo 'machine urs.earthdata.nasa.gov login $user_name password $password' >> $cookies
