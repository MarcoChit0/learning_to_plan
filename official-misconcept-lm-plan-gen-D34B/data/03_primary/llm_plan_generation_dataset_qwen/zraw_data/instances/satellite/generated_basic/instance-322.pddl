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
	spectrograph0 - mode
	spectrograph1 - mode
	image2 - mode
	spectrograph3 - mode
	star4 - direction
	groundstation5 - direction
	star2 - direction
	star0 - direction
	star1 - direction
	star3 - direction
	star6 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 spectrograph1)
	(supports instrument3 image2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star3)
	(supports instrument5 image2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(have_image star6 image2)
))

)
