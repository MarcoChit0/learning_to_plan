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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared1 - mode
	spectrograph0 - mode
	star4 - direction
	groundstation0 - direction
	star3 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation2)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
)
(:goal (and
	(pointing satellite1 star3)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
))

)
