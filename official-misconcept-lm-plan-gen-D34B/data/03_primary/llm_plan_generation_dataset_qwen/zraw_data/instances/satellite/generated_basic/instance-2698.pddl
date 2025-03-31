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
	image2 - mode
	spectrograph3 - mode
	spectrograph1 - mode
	image0 - mode
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph3)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image planet4 image0)
))

)
