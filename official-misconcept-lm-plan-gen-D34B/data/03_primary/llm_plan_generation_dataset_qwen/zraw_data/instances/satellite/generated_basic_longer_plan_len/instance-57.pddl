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
	instrument7 - instrument
	image0 - mode
	spectrograph1 - mode
	image2 - mode
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	planet3 - direction
	phenomenon4 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(supports instrument6 image2)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph1)
	(supports instrument7 image2)
	(supports instrument7 image0)
	(calibration_target instrument7 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite1 star2)
	(pointing satellite2 phenomenon6)
	(have_image planet3 spectrograph1)
	(have_image phenomenon4 image0)
	(have_image star5 image2)
	(have_image phenomenon6 spectrograph1)
	(have_image star7 spectrograph1)
))

)
