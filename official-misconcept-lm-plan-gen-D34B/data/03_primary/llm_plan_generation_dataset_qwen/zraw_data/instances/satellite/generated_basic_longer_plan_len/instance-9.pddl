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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared3 - mode
	spectrograph1 - mode
	image0 - mode
	spectrograph2 - mode
	star1 - direction
	star3 - direction
	star2 - direction
	groundstation0 - direction
	groundstation4 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared3)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared3)
	(supports instrument8 spectrograph1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star3)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star3)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument12 image0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared3)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 groundstation4)
	(supports instrument14 image0)
	(supports instrument14 infrared3)
	(calibration_target instrument14 star2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument15 spectrograph1)
	(supports instrument15 infrared3)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 spectrograph2)
	(supports instrument16 infrared3)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 image0)
	(supports instrument17 infrared3)
	(calibration_target instrument17 groundstation4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite1 planet10)
	(have_image star5 infrared3)
	(have_image phenomenon6 image0)
	(have_image phenomenon7 spectrograph2)
	(have_image phenomenon8 spectrograph2)
	(have_image phenomenon9 spectrograph2)
	(have_image planet10 image0)
))

)
