(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	spectrograph3 - mode
	image1 - mode
	infrared2 - mode
	infrared0 - mode
	star2 - direction
	star1 - direction
	groundstation3 - direction
	groundstation0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared0)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph3)
	(supports instrument7 infrared2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star1)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared2)
	(supports instrument10 infrared0)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument11 infrared0)
	(supports instrument11 infrared2)
	(supports instrument11 image1)
	(calibration_target instrument11 star2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 image1)
	(supports instrument13 infrared2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph3)
	(supports instrument14 image1)
	(calibration_target instrument14 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 spectrograph3)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation3)
	(supports instrument19 infrared2)
	(calibration_target instrument19 groundstation0)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 groundstation3)
	(pointing satellite3 phenomenon6)
	(pointing satellite4 star2)
	(have_image phenomenon4 infrared2)
	(have_image phenomenon5 infrared2)
	(have_image phenomenon6 image1)
))

)
