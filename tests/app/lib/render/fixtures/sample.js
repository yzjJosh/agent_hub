function greet(name) {
  const msg = `Hello, ${name}!`;
  return msg;
}

for (let i = 0; i < 3; i += 1) {
  console.log(greet(`user${i}`));
}
