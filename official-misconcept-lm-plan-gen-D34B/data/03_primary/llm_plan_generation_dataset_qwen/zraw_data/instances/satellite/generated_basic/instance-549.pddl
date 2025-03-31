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
	satellite2 - satellite
	instrument6 - instrument
	image2 - mode
	spectrograph1 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image2)
	(supports instrument4 spectrograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 image2)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
)
(:goal (and
	(pointing satellite2 planet2)
	(have_image planet2 spectrograph1)
	(have_image star3 image0)
	(have_image planet4 image2)
	(have_image star5 image2)
	(have_image star6 spectrograph1)
))

)
