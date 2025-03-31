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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph3 - mode
	image1 - mode
	image2 - mode
	thermograph0 - mode
	star3 - direction
	groundstation2 - direction
	star0 - direction
	star1 - direction
	groundstation4 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 image1)
	(supports instrument3 spectrograph3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image1)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star1)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument10 spectrograph3)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument12 image2)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 star1)
	(supports instrument13 image1)
	(supports instrument13 image2)
	(calibration_target instrument13 star0)
	(supports instrument14 image2)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 star1)
	(supports instrument15 image1)
	(supports instrument15 spectrograph3)
	(supports instrument15 image2)
	(calibration_target instrument15 groundstation4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation4)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite2 star3)
	(have_image phenomenon5 image1)
	(have_image planet6 thermograph0)
	(have_image star7 image1)
	(have_image phenomenon8 thermograph0)
	(have_image phenomenon9 image1)
	(have_image phenomenon10 spectrograph3)
))

)
