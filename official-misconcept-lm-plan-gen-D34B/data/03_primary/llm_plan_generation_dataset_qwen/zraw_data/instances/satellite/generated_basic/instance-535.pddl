(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	thermograph0 - mode
	thermograph3 - mode
	image1 - mode
	image2 - mode
	groundstation1 - direction
	groundstation3 - direction
	star4 - direction
	star2 - direction
	star0 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 image1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
)
(:goal (and
	(pointing satellite0 star8)
	(pointing satellite1 phenomenon5)
	(have_image phenomenon5 thermograph0)
	(have_image planet6 thermograph0)
	(have_image planet7 image2)
	(have_image star8 image2)
))

)
