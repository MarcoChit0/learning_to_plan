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
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	thermograph0 - mode
	infrared2 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 image1)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared2)
	(supports instrument6 image1)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared2)
	(supports instrument7 image1)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared2)
	(supports instrument8 image1)
	(calibration_target instrument8 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image1)
	(supports instrument10 infrared2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star2)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 image1)
	(calibration_target instrument12 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
)
(:goal (and
	(have_image planet3 thermograph0)
	(have_image planet4 thermograph0)
	(have_image phenomenon5 infrared2)
	(have_image star6 thermograph0)
))

)
