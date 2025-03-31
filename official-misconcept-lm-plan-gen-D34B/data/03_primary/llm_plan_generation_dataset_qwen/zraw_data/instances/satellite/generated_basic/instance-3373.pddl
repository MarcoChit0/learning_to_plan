(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image3 - mode
	image1 - mode
	infrared0 - mode
	spectrograph2 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image3)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 spectrograph2)
	(supports instrument5 image3)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph2)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(supports instrument7 image3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite2 groundstation1)
	(have_image star2 image1)
	(have_image planet3 spectrograph2)
))

)
