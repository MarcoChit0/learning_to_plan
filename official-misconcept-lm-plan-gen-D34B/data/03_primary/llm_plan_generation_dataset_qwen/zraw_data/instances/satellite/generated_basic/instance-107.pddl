(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	infrared0 - mode
	spectrograph1 - mode
	groundstation4 - direction
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star0 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star5)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 spectrograph1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument10 spectrograph1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star5)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 groundstation4)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation3)
	(calibration_target instrument13 star5)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 star1)
	(supports instrument15 spectrograph1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star5)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument16 spectrograph1)
	(calibration_target instrument16 groundstation3)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 spectrograph1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared0)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star5)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image star6 spectrograph1)
	(have_image star7 infrared0)
	(have_image planet8 spectrograph1)
	(have_image phenomenon9 spectrograph1)
))

)
