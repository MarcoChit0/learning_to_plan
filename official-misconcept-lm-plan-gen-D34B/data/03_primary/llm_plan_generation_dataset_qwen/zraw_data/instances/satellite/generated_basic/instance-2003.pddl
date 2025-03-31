(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	spectrograph0 - mode
	image1 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 planet2)
	(have_image planet2 image1)
))

)
