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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph2 - mode
	image1 - mode
	image0 - mode
	thermograph3 - mode
	star0 - direction
	star2 - direction
	groundstation4 - direction
	star3 - direction
	star1 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star3)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 image1)
	(supports instrument4 thermograph3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 image0)
	(supports instrument5 thermograph3)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument8 thermograph3)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 image1)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 spectrograph2)
	(supports instrument11 image0)
	(supports instrument11 image1)
	(calibration_target instrument11 star1)
	(supports instrument12 image1)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
	(supports instrument13 image1)
	(supports instrument13 spectrograph2)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation4)
	(supports instrument14 thermograph3)
	(supports instrument14 image1)
	(calibration_target instrument14 star3)
	(supports instrument15 thermograph3)
	(supports instrument15 image1)
	(supports instrument15 image0)
	(calibration_target instrument15 star1)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon9)
)
(:goal (and
	(have_image phenomenon5 image1)
	(have_image planet6 spectrograph2)
	(have_image star7 image0)
	(have_image planet8 thermograph3)
	(have_image phenomenon9 thermograph3)
	(have_image planet10 thermograph3)
))

)
