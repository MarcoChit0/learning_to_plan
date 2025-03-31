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
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	star5 - direction
	star1 - direction
	star3 - direction
	groundstation0 - direction
	star4 - direction
	groundstation2 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star5)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 star3)
	(supports instrument10 infrared1)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 star4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation0)
	(calibration_target instrument11 star3)
	(supports instrument12 thermograph0)
	(supports instrument12 infrared1)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star5)
	(supports instrument13 infrared1)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star1)
	(calibration_target instrument13 star5)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument14 infrared1)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 star3)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph0)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 thermograph0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared1)
	(calibration_target instrument17 groundstation2)
	(calibration_target instrument17 star4)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet9)
)
(:goal (and
	(pointing satellite2 phenomenon7)
	(have_image star6 infrared1)
	(have_image phenomenon7 thermograph0)
	(have_image planet8 thermograph0)
	(have_image planet9 spectrograph2)
))

)
