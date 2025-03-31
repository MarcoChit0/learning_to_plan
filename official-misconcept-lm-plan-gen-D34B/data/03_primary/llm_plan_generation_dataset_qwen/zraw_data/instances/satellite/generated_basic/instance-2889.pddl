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
	satellite3 - satellite
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph0 - mode
	groundstation5 - direction
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation1 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation4)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation1)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star0)
	(calibration_target instrument11 groundstation5)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation5)
	(calibration_target instrument12 groundstation4)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation3)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star0)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation4)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation4)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 groundstation1)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite1 planet7)
	(pointing satellite2 planet7)
	(have_image planet6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image star8 spectrograph0)
))

)
