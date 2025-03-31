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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	spectrograph3 - mode
	spectrograph1 - mode
	thermograph0 - mode
	image2 - mode
	star2 - direction
	groundstation1 - direction
	groundstation0 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph3)
	(supports instrument4 image2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star2)
	(supports instrument7 image2)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument9 image2)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation1)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument10 spectrograph1)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph3)
	(supports instrument11 image2)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image phenomenon4 image2)
	(have_image star5 spectrograph3)
	(have_image planet6 thermograph0)
))

)
