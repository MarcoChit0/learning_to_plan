(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image3 - mode
	infrared1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	groundstation4 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	star2 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument4 spectrograph2)
	(supports instrument4 image3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared1)
	(supports instrument5 image3)
	(calibration_target instrument5 star2)
	(supports instrument6 image3)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument8 image3)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image3)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 spectrograph2)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star0)
	(supports instrument12 spectrograph2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star3)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(have_image star5 spectrograph2)
	(have_image phenomenon6 thermograph0)
))

)
