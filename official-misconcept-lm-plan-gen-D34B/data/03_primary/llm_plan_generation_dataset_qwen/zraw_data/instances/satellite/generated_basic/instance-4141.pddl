(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image1 - mode
	spectrograph0 - mode
	groundstation1 - direction
	star3 - direction
	groundstation5 - direction
	star4 - direction
	groundstation0 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image1)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 image1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star4)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star4)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 spectrograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation2)
	(calibration_target instrument13 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(have_image star6 spectrograph0)
))

)
