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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	image2 - mode
	spectrograph1 - mode
	infrared3 - mode
	infrared0 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument5 spectrograph1)
	(supports instrument5 image2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon3)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image planet1 spectrograph1)
	(have_image phenomenon2 image2)
	(have_image phenomenon3 image2)
))

)
