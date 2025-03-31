(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image2 - mode
	thermograph1 - mode
	thermograph0 - mode
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 image2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 thermograph0)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph1)
	(supports instrument8 thermograph0)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image2)
	(supports instrument12 thermograph0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 thermograph1)
	(supports instrument13 image2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite2 star2)
	(have_image star2 thermograph0)
	(have_image star3 image2)
))

)
