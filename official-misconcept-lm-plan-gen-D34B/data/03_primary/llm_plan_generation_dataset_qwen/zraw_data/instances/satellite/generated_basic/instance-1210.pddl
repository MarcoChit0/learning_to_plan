(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image0 - mode
	infrared3 - mode
	spectrograph1 - mode
	spectrograph2 - mode
	star0 - direction
	groundstation2 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared3)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared3)
	(supports instrument3 image0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph1)
	(supports instrument8 image0)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
)
(:goal (and
	(have_image star3 spectrograph1)
	(have_image planet4 spectrograph2)
	(have_image planet5 infrared3)
))

)
