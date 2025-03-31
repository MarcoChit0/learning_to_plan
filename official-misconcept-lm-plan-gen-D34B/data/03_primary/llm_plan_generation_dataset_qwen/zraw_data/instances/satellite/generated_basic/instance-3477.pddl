(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	infrared1 - mode
	image0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation1 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite2 planet3)
	(have_image planet3 infrared1)
	(have_image star4 infrared1)
))

)
