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
	image1 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	image0 - mode
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph2)
	(supports instrument5 image0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph3)
	(supports instrument7 image1)
	(calibration_target instrument7 star3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(have_image planet4 spectrograph3)
	(have_image planet5 image0)
	(have_image star6 spectrograph2)
))

)
