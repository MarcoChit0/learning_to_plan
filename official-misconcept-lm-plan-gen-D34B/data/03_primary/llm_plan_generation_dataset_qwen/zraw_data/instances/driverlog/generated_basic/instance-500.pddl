(define (problem DLOG-2-1-1)
	(:domain driverlog)
	(:objects
	driver269 - driver
	driver272 - driver
	truck112 - truck
	package375 - obj
	s0 - location
	s1 - location
	p0-1 - location
	p1-0 - location
	)
	(:init
	(at driver269 s0)
	(at driver272 s1)
	(at truck112 s0)
	(empty truck112)
	(at package375 s1)
	(path s0 p0-1)
	(path p0-1 s0)
	(path s1 p0-1)
	(path p0-1 s1)
	(link s0 s1)
	(link s1 s0)
)
	(:goal (and
	(at driver269 s0)
	(at package375 s0)
	))


)
