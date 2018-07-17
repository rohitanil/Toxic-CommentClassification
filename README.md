# Private URL shortener

A lot of companies often share links on promotional material, and quite often, these links can be long and difficult to type. And displaying a long link on printed media pretty much guarantees that no one will bother to type it in.

Companies can use third-party URL shorteners like bit.ly, but a lot of companies decide to buy a short domain themselves, because custom short URLs are very useful when sharing posts on social media, by preserving company branding.

For example, Verge uses the short domain `vrge.co/<SHORTCODE>` to shorten `theverge.com` URLs.

We'd like you to take a crack at build something similar - a useful feature for most web products.

## Basic requirements

You'll need NodeJS installed on your system to complete these tasks. There are no other requirements. These tasks have been tested with the latest LTS version of Node (v8.11.X).

To install Node, follow [instructions on the official website](https://nodejs.org/en/), or just google how to do it on your platform.

With node installed, you'll need to install all required modules in each task directory before starting work. For example, before starting work on the first task, you'll need to:

    $ cd task1
    $ npm install

With all dependencies installed, check out the `README.md` file in the task folder for more information.

## How to submit your code

Once you have completed the tasks:

1.  Create an archive of all files.
2.  Keep the name of the archive the same as the task bundle you downloaded. For example: `url-shortener-20180717.zip`.
3.  Use the following form to submit the file: https://svlabs.typeform.com/to/oUuwuZ

Let's get started with the tasks, shall we?

## Task 1: Warm up with a function.

`/task1/` contains a simple script with a function named `shorten`, which accepts a string `url` and an integer `length`. The function is empty at the moment. We'd like you to fill in an implementation. See `task1/README.md` for more information.

Required:

1.  The function returns a _shortcode_ of the given length.
2.  The return should only include numbers (0-9) and letters (a-z, A-Z).
3.  Repeated calls with the same URL should return the same _shortcode_. There's no point in generating a new code for the same URL.

Also note:

1.  You can test your code by running the test script in the repository. See the README file for more details.
2.  Submit your solution once you get the tests passing **and** you are happy with how your code looks.

## Task 2: Back-end.

`/task2/` contains a simple Express server that has two end-points:

1.  `POST /shortcodes` to shorten a given URL, and...
2.  `GET /shortcodes/[SHORTCODE]` that returns the URL corresponding to a short-code.

The task contains an _Express_ server ready for you to customize. You'll note that both end-points have empty handler functions.

We've written tests that check for the desired functionality. Writing tests first is often a good way to ensure that what you're building is what you really need.

Required:

1.  Using these tests, fill in the API endpoint handlers so that they function as the tests expect.
2.  Use code from your previous task!

Also note:

1.  Please submit your solution after you get all the tests to pass.
2.  Hint: You might have to edit your previous task for all tests to pass correctly!

## Task 3: `App = frontEnd + backEnd`

`/task3/` contains both an front-end built with React. It contains text inputs, submit buttons, and `div`-s to display generated shortcodes and retrieved URLs. However, submitting the form does nothing right now - that's up to you.

Required:

1.  We'd like you to get the front-end working!
2.  Filling in an input with a URL, and clicking the submit button should display the resulting shortcode in the corresponding output `div`.
3.  Filling in an input with the shortcode, and clicking the submit button should display the original URL in the corresponding output `div`.

Also note:

1.  Tests have available in the repo to check whether your code works as expected.
2.  The README file in the repository includes detailed instructions on how to build and load and the front-end.

## Task 4: Planning for the future.

You don't have to write any code for this task. We would simply like you give your opinions on how some additional requirements can be accommodated. Write a design note on how you'll implement some new features:

Required:

1.  Please edit `task4/DESIGN.md` & fill in the required sections. You can add additional sections if you'd like, but please do not delete any existing sections.

Also note:

1.  If you'd like to provide figures or diagrams, please make sure they are in .PNG format. Push them to the same folder as DESIGN.md.
