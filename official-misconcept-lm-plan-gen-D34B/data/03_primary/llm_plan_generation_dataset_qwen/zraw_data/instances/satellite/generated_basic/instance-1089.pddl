(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	infrared2 - mode
	infrared0 - mode
	infrared1 - mode
	star2 - direction
	groundstation3 - direction
	star0 - direction
	star4 - direction
	star5 - direction
	star1 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star1)
	(calibration_target instrument2 star5)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite0 planet8)
	(have_image planet6 infrared2)
	(have_image star7 infrared0)
	(have_image planet8 infrared2)
))

)
