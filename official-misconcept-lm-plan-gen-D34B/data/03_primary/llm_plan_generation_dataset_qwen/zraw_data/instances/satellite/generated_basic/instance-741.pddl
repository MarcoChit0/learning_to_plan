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
	instrument15 - instrument
	instrument16 - instrument
	spectrograph0 - mode
	thermograph2 - mode
	image1 - mode
	star0 - direction
	groundstation1 - direction
	star5 - direction
	groundstation2 - direction
	star4 - direction
	star3 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 star5)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 star4)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star3)
	(supports instrument5 image1)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star5)
	(supports instrument8 image1)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star4)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(calibration_target instrument9 star5)
	(supports instrument10 image1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 star5)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star5)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 star5)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 spectrograph0)
	(supports instrument15 image1)
	(calibration_target instrument15 star3)
	(calibration_target instrument15 star4)
	(supports instrument16 image1)
	(calibration_target instrument16 star3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image planet6 spectrograph0)
	(have_image star7 thermograph2)
	(have_image phenomenon8 spectrograph0)
	(have_image planet9 thermograph2)
))

)
