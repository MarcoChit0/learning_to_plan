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
	satellite2 - satellite
	instrument7 - instrument
	image2 - mode
	infrared3 - mode
	spectrograph1 - mode
	infrared0 - mode
	star0 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation3 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image phenomenon4 spectrograph1)
	(have_image star5 infrared0)
	(have_image star6 infrared3)
))

)
