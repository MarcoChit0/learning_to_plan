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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	infrared2 - mode
	thermograph1 - mode
	thermograph0 - mode
	image3 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 image3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument6 thermograph0)
	(supports instrument6 image3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph1)
	(supports instrument7 image3)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument8 thermograph1)
	(supports instrument8 thermograph0)
	(supports instrument8 image3)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite2 star0)
	(pointing satellite3 star1)
	(have_image star1 image3)
	(have_image phenomenon2 thermograph0)
	(have_image phenomenon3 thermograph1)
))

)
