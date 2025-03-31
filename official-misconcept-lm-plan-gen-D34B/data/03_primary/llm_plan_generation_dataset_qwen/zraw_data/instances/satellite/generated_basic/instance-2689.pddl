(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared3 - mode
	infrared1 - mode
	thermograph0 - mode
	image2 - mode
	star2 - direction
	groundstation0 - direction
	star1 - direction
	groundstation3 - direction
	star4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 infrared1)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 groundstation3)
	(have_image star4 thermograph0)
))

)
