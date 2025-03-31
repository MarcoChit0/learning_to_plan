(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	image2 - mode
	image1 - mode
	thermograph3 - mode
	image0 - mode
	groundstation3 - direction
	groundstation4 - direction
	star0 - direction
	star1 - direction
	star2 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph3)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(supports instrument4 image2)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(have_image star5 image1)
	(have_image phenomenon6 image2)
	(have_image phenomenon7 image0)
	(have_image planet8 image0)
))

)
