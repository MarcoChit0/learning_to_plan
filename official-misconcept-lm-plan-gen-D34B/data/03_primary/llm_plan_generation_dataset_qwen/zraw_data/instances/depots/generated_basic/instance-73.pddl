(define (problem depot-1-3-6-4-6-3) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 distributor1 distributor2 - Distributor
	truck0 truck1 truck2 truck3 truck4 truck5 - Truck
	pallet0 pallet1 pallet2 pallet3 - Pallet
	crate0 crate1 crate2 - Crate
	hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 - Hoist)
(:init
	(at pallet0 depot0)
	(clear pallet0)
	(at pallet1 distributor0)
	(clear crate1)
	(at pallet2 distributor1)
	(clear crate0)
	(at pallet3 distributor2)
	(clear crate2)
	(at truck0 distributor1)
	(at truck1 distributor2)
	(at truck2 distributor0)
	(at truck3 distributor1)
	(at truck4 distributor2)
	(at truck5 depot0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at hoist2 distributor1)
	(available hoist2)
	(at hoist3 distributor2)
	(available hoist3)
	(at hoist4 distributor2)
	(available hoist4)
	(at hoist5 distributor0)
	(available hoist5)
	(at crate0 distributor1)
	(on crate0 pallet2)
	(at crate1 distributor0)
	(on crate1 pallet1)
	(at crate2 distributor2)
	(on crate2 pallet3)
)

(:goal (and
		(on crate0 pallet0)
		(on crate2 crate0)
	)
))
