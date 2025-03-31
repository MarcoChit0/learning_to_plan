(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	infrared1 - mode
	image2 - mode
	spectrograph3 - mode
	infrared0 - mode
	star4 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star0 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared0)
	(supports instrument6 image2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument8 infrared1)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite0 planet5)
	(have_image planet5 infrared1)
	(have_image phenomenon6 spectrograph3)
	(have_image phenomenon7 infrared0)
))

)
