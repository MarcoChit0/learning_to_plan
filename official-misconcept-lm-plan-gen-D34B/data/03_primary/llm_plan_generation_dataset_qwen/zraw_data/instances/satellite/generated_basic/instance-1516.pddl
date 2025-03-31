(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	infrared0 - mode
	spectrograph1 - mode
	star3 - direction
	groundstation4 - direction
	star1 - direction
	star2 - direction
	groundstation0 - direction
	groundstation5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation0)
	(calibration_target instrument7 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(calibration_target instrument8 star2)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star1)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 infrared0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star3)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 groundstation5)
	(calibration_target instrument12 star3)
	(supports instrument13 infrared0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 groundstation4)
	(calibration_target instrument13 groundstation5)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 star1)
	(calibration_target instrument14 groundstation5)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared0)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 groundstation5)
	(supports instrument18 spectrograph1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 infrared0)
	(supports instrument19 spectrograph1)
	(calibration_target instrument19 groundstation5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation4)
)
(:goal (and
	(pointing satellite1 groundstation4)
	(pointing satellite3 groundstation5)
	(have_image phenomenon6 spectrograph1)
	(have_image phenomenon7 infrared0)
	(have_image planet8 infrared0)
))

)
