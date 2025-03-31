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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	image3 - mode
	spectrograph2 - mode
	image1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 spectrograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(pointing satellite2 groundstation0)
	(have_image star3 image0)
	(have_image planet4 image1)
))

)
