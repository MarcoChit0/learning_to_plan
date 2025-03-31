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
	satellite2 - satellite
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	infrared0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument8 spectrograph1)
	(supports instrument8 infrared0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation3)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument11 spectrograph2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(supports instrument12 infrared0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
)
(:goal (and
	(pointing satellite3 groundstation1)
	(have_image phenomenon4 infrared0)
	(have_image phenomenon5 infrared0)
	(have_image star6 infrared0)
	(have_image planet7 infrared0)
	(have_image planet8 spectrograph2)
	(have_image planet9 spectrograph2)
))

)
