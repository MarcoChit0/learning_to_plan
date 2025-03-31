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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph1 - mode
	image3 - mode
	spectrograph0 - mode
	spectrograph2 - mode
	groundstation1 - direction
	groundstation2 - direction
	star0 - direction
	groundstation3 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph1)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument6 image3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 image3)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 thermograph1)
	(supports instrument8 image3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph2)
	(supports instrument10 image3)
	(calibration_target instrument10 groundstation3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite1 planet4)
	(have_image planet4 thermograph1)
	(have_image star5 image3)
	(have_image phenomenon6 spectrograph0)
))

)
