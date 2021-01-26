package main

import (
	"fmt"
	"time"
)

type SlowAddPromise struct {
	c chan int
}

// NewSlowAddPromise returns a number slowly
func NewSlowAddPromise(n int, delay time.Duration) *SlowAddPromise {
	p := SlowAddPromise{make(chan int)}
	go func() {
		time.Sleep(delay)
		p.c <- n + 1
		close(p.c)
	}()
	return &p
}

// Evaluate evaluates the promise
func (promise *SlowAddPromise) Evaluate() int {
	return <-promise.c
}

func main() {
	c := make(chan *SlowAddPromise, 10)
	c <- NewSlowAddPromise(5, 1*time.Second)
	c <- NewSlowAddPromise(9, 2*time.Second)
	c <- NewSlowAddPromise(68, 2*time.Second)
	close(c)

	for p := range c {
		fmt.Println(p.Evaluate())
	}

	fmt.Println("x_x")
}
