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
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	spectrograph1 - mode
	spectrograph2 - mode
	infrared3 - mode
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation3 - direction
	star2 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared3)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 infrared3)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument11 image0)
	(supports instrument11 infrared3)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared3)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 groundstation1)
	(pointing satellite2 star5)
	(have_image phenomenon4 spectrograph2)
	(have_image star5 spectrograph2)
	(have_image star6 image0)
))

)
