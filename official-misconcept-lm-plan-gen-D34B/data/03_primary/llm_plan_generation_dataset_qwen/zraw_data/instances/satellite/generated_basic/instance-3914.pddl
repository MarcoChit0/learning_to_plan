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
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	image2 - mode
	infrared0 - mode
	image1 - mode
	spectrograph3 - mode
	groundstation4 - direction
	groundstation3 - direction
	star1 - direction
	groundstation5 - direction
	groundstation2 - direction
	groundstation0 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument3 spectrograph3)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 spectrograph3)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument8 image2)
	(supports instrument8 spectrograph3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph3)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image2)
	(supports instrument10 spectrograph3)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph3)
	(supports instrument11 infrared0)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation5)
	(supports instrument12 image2)
	(supports instrument12 image1)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 groundstation2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument13 spectrograph3)
	(supports instrument13 image2)
	(supports instrument13 image1)
	(calibration_target instrument13 star1)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
	(supports instrument14 image2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(calibration_target instrument14 groundstation4)
	(supports instrument15 infrared0)
	(supports instrument15 image2)
	(calibration_target instrument15 star1)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation5)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared0)
	(supports instrument17 spectrograph3)
	(calibration_target instrument17 groundstation0)
	(calibration_target instrument17 groundstation2)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(pointing satellite2 phenomenon7)
	(pointing satellite3 star1)
	(have_image phenomenon6 spectrograph3)
	(have_image phenomenon7 spectrograph3)
	(have_image phenomenon8 image2)
))

)
