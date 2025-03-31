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
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	thermograph0 - mode
	image3 - mode
	spectrograph1 - mode
	image2 - mode
	groundstation0 - direction
	star4 - direction
	star2 - direction
	star1 - direction
	star5 - direction
	star3 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	phenomenon10 - direction
	planet11 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph0)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 image3)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star5)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image2)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 star2)
	(supports instrument7 image3)
	(supports instrument7 image2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon9)
	(supports instrument9 image3)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 star4)
	(supports instrument10 image2)
	(calibration_target instrument10 star3)
	(calibration_target instrument10 star5)
	(supports instrument11 thermograph0)
	(supports instrument11 spectrograph1)
	(supports instrument11 image2)
	(calibration_target instrument11 star2)
	(supports instrument12 image3)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star5)
	(calibration_target instrument12 star2)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet11)
	(supports instrument13 thermograph0)
	(supports instrument13 image2)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star1)
	(calibration_target instrument13 star5)
	(supports instrument14 image2)
	(calibration_target instrument14 star3)
	(calibration_target instrument14 star5)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon10)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(pointing satellite2 star7)
	(have_image star6 image2)
	(have_image star7 image2)
	(have_image phenomenon8 image2)
	(have_image phenomenon9 image3)
	(have_image phenomenon10 image3)
	(have_image planet11 thermograph0)
))

)
