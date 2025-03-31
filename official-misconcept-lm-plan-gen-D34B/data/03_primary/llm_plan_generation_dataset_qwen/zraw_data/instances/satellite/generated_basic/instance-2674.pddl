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
	spectrograph3 - mode
	thermograph2 - mode
	spectrograph1 - mode
	image0 - mode
	star2 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 image0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 spectrograph3)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph3)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image phenomenon4 image0)
))

)
