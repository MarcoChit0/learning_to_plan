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
	spectrograph3 - mode
	infrared0 - mode
	infrared1 - mode
	image2 - mode
	star1 - direction
	groundstation3 - direction
	star0 - direction
	star2 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph3)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image star4 spectrograph3)
))

)
