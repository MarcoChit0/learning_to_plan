(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	spectrograph0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation4 - direction
	star1 - direction
	star3 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation4)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star3)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 star3)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite2 groundstation4)
	(pointing satellite5 groundstation4)
	(have_image planet5 spectrograph0)
	(have_image phenomenon6 spectrograph0)
))

)
