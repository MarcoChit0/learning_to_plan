(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	spectrograph0 - mode
	star2 - direction
	star5 - direction
	groundstation3 - direction
	groundstation0 - direction
	star1 - direction
	groundstation4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation4)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image phenomenon6 spectrograph0)
))

)
