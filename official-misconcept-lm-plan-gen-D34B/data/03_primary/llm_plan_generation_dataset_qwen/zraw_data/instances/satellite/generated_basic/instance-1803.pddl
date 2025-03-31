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
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	spectrograph1 - mode
	infrared2 - mode
	thermograph0 - mode
	spectrograph3 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph0)
	(supports instrument7 spectrograph3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared2)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph0)
	(supports instrument10 spectrograph1)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument11 spectrograph3)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph1)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation0)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(have_image planet3 spectrograph3)
	(have_image star4 infrared2)
))

)
