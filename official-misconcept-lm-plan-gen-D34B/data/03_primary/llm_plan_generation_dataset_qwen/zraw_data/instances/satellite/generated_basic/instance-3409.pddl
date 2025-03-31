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
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	spectrograph0 - mode
	infrared1 - mode
	image2 - mode
	star1 - direction
	star0 - direction
	star3 - direction
	groundstation2 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared1)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument4 spectrograph0)
	(supports instrument4 image2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 image2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 image2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph0)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 planet4)
	(have_image planet4 image2)
))

)
