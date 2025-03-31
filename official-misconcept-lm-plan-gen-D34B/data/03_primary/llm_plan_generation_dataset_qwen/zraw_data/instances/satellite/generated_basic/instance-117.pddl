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
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph1 - mode
	spectrograph0 - mode
	star3 - direction
	star0 - direction
	star2 - direction
	star1 - direction
	star5 - direction
	star4 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star4)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star4)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star4)
	(calibration_target instrument10 star1)
	(supports instrument11 spectrograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 star5)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument14 spectrograph0)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 star2)
	(supports instrument15 spectrograph0)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 star4)
	(supports instrument16 spectrograph0)
	(supports instrument16 spectrograph1)
	(calibration_target instrument16 star2)
	(calibration_target instrument16 star4)
	(supports instrument17 spectrograph0)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 star5)
	(calibration_target instrument17 star1)
	(supports instrument18 spectrograph1)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite3 planet6)
	(pointing satellite4 star3)
	(have_image planet6 spectrograph1)
	(have_image star7 spectrograph1)
	(have_image planet8 spectrograph0)
	(have_image planet9 spectrograph0)
))

)
