(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	image0 - mode
	thermograph1 - mode
	image2 - mode
	groundstation0 - direction
	phenomenon1 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 thermograph1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon1)
	(supports instrument8 image2)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image0)
	(supports instrument9 image2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 phenomenon1)
	(pointing satellite2 phenomenon1)
	(pointing satellite3 phenomenon1)
	(have_image phenomenon1 thermograph1)
))

)
