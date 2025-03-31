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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph0 - mode
	spectrograph3 - mode
	thermograph2 - mode
	infrared1 - mode
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument8 spectrograph3)
	(supports instrument8 thermograph0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(have_image phenomenon3 spectrograph3)
	(have_image phenomenon4 infrared1)
))

)
