(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph1 - mode
	image0 - mode
	star3 - direction
	star1 - direction
	star0 - direction
	star2 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star2)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 star4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite2 star0)
	(have_image phenomenon5 spectrograph1)
))

)
