(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	image0 - mode
	thermograph2 - mode
	image1 - mode
	groundstation0 - direction
	star1 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image star1 image1)
))

)
