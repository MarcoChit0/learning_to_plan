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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	spectrograph1 - mode
	image3 - mode
	infrared0 - mode
	spectrograph2 - mode
	groundstation0 - direction
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 image3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 image3)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument6 infrared0)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star9)
	(supports instrument8 infrared0)
	(supports instrument8 spectrograph2)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image3)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
)
(:goal (and
	(pointing satellite3 groundstation1)
	(have_image planet4 infrared0)
	(have_image planet5 image3)
	(have_image star6 image3)
	(have_image star7 infrared0)
	(have_image phenomenon8 spectrograph1)
	(have_image star9 image3)
))

)
