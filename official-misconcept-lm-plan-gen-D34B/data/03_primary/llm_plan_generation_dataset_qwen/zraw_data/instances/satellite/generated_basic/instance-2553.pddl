(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared2 - mode
	image1 - mode
	thermograph0 - mode
	star2 - direction
	groundstation3 - direction
	groundstation0 - direction
	star4 - direction
	star1 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image planet5 image1)
))

)
