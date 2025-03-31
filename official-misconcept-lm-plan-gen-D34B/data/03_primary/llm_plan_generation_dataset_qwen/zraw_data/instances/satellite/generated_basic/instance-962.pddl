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
	satellite2 - satellite
	instrument7 - instrument
	image0 - mode
	spectrograph3 - mode
	image2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star3 - direction
	star2 - direction
	star1 - direction
	phenomenon4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph1)
	(supports instrument1 image2)
	(supports instrument1 image0)
	(calibration_target instrument1 star3)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 image2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 phenomenon6)
	(pointing satellite2 star2)
	(have_image phenomenon4 spectrograph3)
	(have_image star5 spectrograph3)
	(have_image phenomenon6 image0)
))

)
