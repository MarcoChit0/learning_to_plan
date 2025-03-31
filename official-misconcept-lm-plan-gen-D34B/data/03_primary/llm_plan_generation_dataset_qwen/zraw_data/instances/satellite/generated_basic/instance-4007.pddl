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
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image2 - mode
	infrared3 - mode
	spectrograph0 - mode
	image1 - mode
	groundstation0 - direction
	groundstation1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image1)
	(supports instrument6 image2)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(supports instrument8 infrared3)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument10 spectrograph0)
	(supports instrument10 image1)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph0)
	(supports instrument11 image1)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image1)
	(supports instrument12 infrared3)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image1)
	(supports instrument14 infrared3)
	(calibration_target instrument14 groundstation1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument15 spectrograph0)
	(supports instrument15 image2)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 image2)
	(calibration_target instrument16 groundstation0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
	(supports instrument17 infrared3)
	(supports instrument17 image2)
	(supports instrument17 spectrograph0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 infrared3)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 image2)
	(supports instrument19 infrared3)
	(calibration_target instrument19 groundstation1)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon2)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(have_image phenomenon2 infrared3)
))

)
