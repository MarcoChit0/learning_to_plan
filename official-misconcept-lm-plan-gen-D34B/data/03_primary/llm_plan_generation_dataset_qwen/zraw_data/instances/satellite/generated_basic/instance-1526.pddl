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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	image1 - mode
	spectrograph0 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	star3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(supports instrument5 image1)
	(calibration_target instrument5 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
	(supports instrument9 image1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star3)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image1)
	(calibration_target instrument12 star4)
	(supports instrument13 image1)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star4)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
	(supports instrument15 image1)
	(calibration_target instrument15 star3)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph0)
	(supports instrument16 image1)
	(calibration_target instrument16 star4)
	(calibration_target instrument16 star3)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
	(supports instrument17 image1)
	(calibration_target instrument17 star5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet6)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite3 star4)
	(pointing satellite5 star4)
	(have_image planet6 spectrograph0)
	(have_image phenomenon7 image1)
	(have_image planet8 spectrograph0)
))

)
