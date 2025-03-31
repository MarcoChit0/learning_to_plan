(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	spectrograph0 - mode
	groundstation2 - direction
	star5 - direction
	groundstation1 - direction
	star0 - direction
	groundstation4 - direction
	star3 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star5)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation4)
	(calibration_target instrument10 star3)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation4)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star3)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star5)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 groundstation1)
	(pointing satellite3 groundstation1)
	(have_image star6 spectrograph0)
))

)
