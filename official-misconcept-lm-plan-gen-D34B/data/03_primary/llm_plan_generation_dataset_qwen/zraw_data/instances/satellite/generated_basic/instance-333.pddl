(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	thermograph0 - mode
	image2 - mode
	thermograph3 - mode
	image1 - mode
	star2 - direction
	groundstation1 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation0 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star5)
	(supports instrument1 image1)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph3)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon6 image2)
))

)
