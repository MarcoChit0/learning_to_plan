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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	infrared1 - mode
	spectrograph0 - mode
	thermograph3 - mode
	thermograph2 - mode
	star2 - direction
	star5 - direction
	groundstation1 - direction
	groundstation4 - direction
	star3 - direction
	star0 - direction
	star6 - direction
	star7 - direction
	star8 - direction
	phenomenon9 - direction
	star10 - direction
	planet11 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph3)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star5)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 star5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph2)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 groundstation4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument9 thermograph3)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph3)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 thermograph3)
	(supports instrument11 spectrograph0)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 thermograph3)
	(supports instrument12 thermograph2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star10)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star0)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star8)
)
(:goal (and
	(pointing satellite0 phenomenon9)
	(have_image star6 infrared1)
	(have_image star7 spectrograph0)
	(have_image star8 thermograph2)
	(have_image phenomenon9 thermograph2)
	(have_image star10 thermograph2)
	(have_image planet11 infrared1)
))

)
