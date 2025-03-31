(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph2 - mode
	image1 - mode
	image0 - mode
	star1 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph2)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument8 thermograph2)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph2)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(supports instrument10 image0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image1)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image star3 image0)
))

)
