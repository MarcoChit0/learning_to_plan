(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared2 - mode
	thermograph3 - mode
	infrared1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	star1 - direction
	planet4 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image planet4 thermograph3)
))

)
