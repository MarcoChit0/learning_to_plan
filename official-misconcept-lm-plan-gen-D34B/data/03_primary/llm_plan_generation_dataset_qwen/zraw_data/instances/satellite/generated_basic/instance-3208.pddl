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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	star1 - direction
	groundstation0 - direction
	star3 - direction
	star2 - direction
	groundstation4 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star5)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument6 thermograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 star5)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument8 image1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 star3)
	(supports instrument9 image1)
	(calibration_target instrument9 star2)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 image1)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star5)
	(calibration_target instrument10 groundstation4)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(have_image planet6 image1)
	(have_image planet7 spectrograph0)
	(have_image phenomenon8 thermograph2)
	(have_image planet9 thermograph2)
))

)
