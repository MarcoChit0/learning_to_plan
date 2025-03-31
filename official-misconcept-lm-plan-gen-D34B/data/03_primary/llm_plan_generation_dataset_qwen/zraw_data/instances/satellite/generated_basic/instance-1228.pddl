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
	image0 - mode
	spectrograph1 - mode
	infrared2 - mode
	infrared3 - mode
	groundstation1 - direction
	star2 - direction
	star0 - direction
	phenomenon3 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared3)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 infrared3)
	(supports instrument5 infrared2)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(pointing satellite1 star5)
	(have_image phenomenon3 infrared2)
	(have_image phenomenon4 image0)
	(have_image star5 spectrograph1)
))

)
