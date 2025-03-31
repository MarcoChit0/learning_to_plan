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
	instrument7 - instrument
	infrared3 - mode
	spectrograph1 - mode
	spectrograph2 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared3)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared3)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 planet4)
	(have_image star3 infrared3)
	(have_image planet4 spectrograph1)
))

)
