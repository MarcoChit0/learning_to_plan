(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	image1 - mode
	image0 - mode
	thermograph2 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	groundstation3 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image phenomenon5 image0)
	(have_image planet6 thermograph2)
	(have_image phenomenon7 image0)
	(have_image phenomenon8 image1)
	(have_image planet9 image1)
))

)
