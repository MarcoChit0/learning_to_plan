(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
)
(:goal (and
	(pointing satellite2 planet1)
	(have_image planet1 spectrograph2)
	(have_image phenomenon2 thermograph0)
))

)
