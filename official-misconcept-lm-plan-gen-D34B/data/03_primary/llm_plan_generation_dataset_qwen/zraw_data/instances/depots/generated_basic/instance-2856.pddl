(define (problem depot-1-3-6-6-4-3) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 distributor1 distributor2 - Distributor
	truck0 truck1 truck2 truck3 truck4 truck5 - Truck
	pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 - Pallet
	crate0 crate1 crate2 - Crate
	hoist0 hoist1 hoist2 hoist3 - Hoist)
(:init
	(at pallet0 depot0)
	(clear pallet0)
	(at pallet1 distributor0)
	(clear pallet1)
	(at pallet2 distributor1)
	(clear pallet2)
	(at pallet3 distributor2)
	(clear crate1)
	(at pallet4 distributor2)
	(clear crate2)
	(at pallet5 distributor1)
	(clear crate0)
	(at truck0 depot0)
	(at truck1 distributor0)
	(at truck2 distributor2)
	(at truck3 distributor1)
	(at truck4 distributor0)
	(at truck5 distributor1)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at hoist2 distributor1)
	(available hoist2)
	(at hoist3 distributor2)
	(available hoist3)
	(at crate0 distributor1)
	(on crate0 pallet5)
	(at crate1 distributor2)
	(on crate1 pallet3)
	(at crate2 distributor2)
	(on crate2 pallet4)
)

(:goal (and
		(on crate0 crate2)
		(on crate1 pallet4)
		(on crate2 pallet0)
	)
))
