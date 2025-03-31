(define (problem depot-1-4-3-7-5-3) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 distributor1 distributor2 distributor3 - Distributor
	truck0 truck1 truck2 - Truck
	pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 - Pallet
	crate0 crate1 crate2 - Crate
	hoist0 hoist1 hoist2 hoist3 hoist4 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate2)
	(at pallet1 distributor0)
	(clear pallet1)
	(at pallet2 distributor1)
	(clear crate1)
	(at pallet3 distributor2)
	(clear pallet3)
	(at pallet4 distributor3)
	(clear pallet4)
	(at pallet5 depot0)
	(clear pallet5)
	(at pallet6 distributor3)
	(clear pallet6)
	(at truck0 distributor3)
	(at truck1 distributor2)
	(at truck2 distributor3)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at hoist2 distributor1)
	(available hoist2)
	(at hoist3 distributor2)
	(available hoist3)
	(at hoist4 distributor3)
	(available hoist4)
	(at crate0 distributor1)
	(on crate0 pallet2)
	(at crate1 distributor1)
	(on crate1 crate0)
	(at crate2 depot0)
	(on crate2 pallet0)
)

(:goal (and
		(on crate1 pallet5)
		(on crate2 pallet2)
	)
))
