(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	image2 - mode
	thermograph1 - mode
	image0 - mode
	star1 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation0 - direction
	star2 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(have_image star5 image0)
	(have_image phenomenon6 thermograph1)
	(have_image star7 image2)
	(have_image phenomenon8 image0)
	(have_image star9 image2)
))

)
