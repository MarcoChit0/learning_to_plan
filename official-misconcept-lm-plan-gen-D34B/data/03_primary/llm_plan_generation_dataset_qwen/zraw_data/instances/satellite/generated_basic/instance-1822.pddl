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
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	infrared3 - mode
	infrared2 - mode
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared3)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared2)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph0)
	(supports instrument11 infrared3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared3)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite1 star4)
	(have_image star3 infrared3)
	(have_image star4 infrared3)
))

)
