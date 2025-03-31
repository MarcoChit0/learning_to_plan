(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	image0 - mode
	thermograph1 - mode
	star3 - direction
	groundstation4 - direction
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite2 planet8)
	(pointing satellite4 phenomenon6)
	(have_image star5 thermograph1)
	(have_image phenomenon6 image0)
	(have_image planet7 thermograph1)
	(have_image planet8 thermograph1)
))

)
