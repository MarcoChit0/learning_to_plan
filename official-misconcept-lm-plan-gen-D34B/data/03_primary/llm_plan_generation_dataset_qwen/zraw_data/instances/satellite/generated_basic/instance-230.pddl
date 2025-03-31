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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph0 - mode
	infrared1 - mode
	groundstation3 - direction
	star1 - direction
	groundstation4 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 infrared1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 spectrograph0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 infrared1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument15 infrared1)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 infrared1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star0)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
)
(:goal (and
	(pointing satellite0 star7)
	(pointing satellite1 star6)
	(pointing satellite3 star7)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 infrared1)
	(have_image star7 infrared1)
	(have_image star8 spectrograph0)
))

)
