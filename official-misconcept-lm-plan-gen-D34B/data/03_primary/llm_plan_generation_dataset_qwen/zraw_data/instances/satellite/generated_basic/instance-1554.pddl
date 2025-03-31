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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation0 - direction
	star1 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image2)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument8 image2)
	(supports instrument8 thermograph1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument9 thermograph1)
	(supports instrument9 image2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph1)
	(supports instrument12 image2)
	(calibration_target instrument12 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image star1 thermograph1)
))

)
