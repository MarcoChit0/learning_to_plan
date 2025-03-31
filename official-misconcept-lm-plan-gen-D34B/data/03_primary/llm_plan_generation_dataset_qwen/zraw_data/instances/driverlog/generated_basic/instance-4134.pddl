(define (problem DLOG-1-1-1)
	(:domain driverlog)
	(:objects
	driver1 - driver
	truck1 - truck
	package1 - obj
	s0 - location
	s1 - location
	s2 - location
	p0-1 - location
	p1-2 - location
	p2-1 - location
	)
	(:init
	(at driver1 s2)
	(at truck1 s2)
	(empty truck1)
	(at package1 s1)
	(path s0 p0-1)
	(path p0-1 s0)
	(path s1 p0-1)
	(path p0-1 s1)
	(path s1 p1-2)
	(path p1-2 s1)
	(path s2 p1-2)
	(path p1-2 s2)
	(link s1 s0)
	(link s0 s1)
	(link s1 s2)
	(link s2 s1)
)
	(:goal (and
	(at driver1 s0)
	(at package1 s1)
	))


)
