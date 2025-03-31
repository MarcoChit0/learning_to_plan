(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph3 - mode
	image2 - mode
	image0 - mode
	infrared1 - mode
	star0 - direction
	planet1 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 image0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(supports instrument6 image2)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 star3)
	(pointing satellite2 planet1)
	(have_image planet1 spectrograph3)
	(have_image star2 spectrograph3)
	(have_image star3 image0)
))

)
