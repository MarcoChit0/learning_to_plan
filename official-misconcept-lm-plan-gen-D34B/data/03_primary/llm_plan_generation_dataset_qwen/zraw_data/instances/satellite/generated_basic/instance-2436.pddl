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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	spectrograph2 - mode
	infrared0 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(supports instrument8 infrared0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(have_image phenomenon2 infrared0)
	(have_image phenomenon3 infrared0)
	(have_image star4 spectrograph2)
	(have_image phenomenon5 infrared0)
))

)
