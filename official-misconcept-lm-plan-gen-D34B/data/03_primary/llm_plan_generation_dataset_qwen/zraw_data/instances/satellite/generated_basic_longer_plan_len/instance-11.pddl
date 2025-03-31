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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	image2 - mode
	image3 - mode
	star4 - direction
	star1 - direction
	groundstation3 - direction
	star2 - direction
	groundstation0 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
	star10 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 image2)
	(calibration_target instrument4 star2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph0)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 spectrograph1)
	(supports instrument9 image3)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star10)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star2)
	(supports instrument11 image2)
	(supports instrument11 spectrograph1)
	(supports instrument11 image3)
	(calibration_target instrument11 groundstation0)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet9)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image planet5 image2)
	(have_image planet6 spectrograph1)
	(have_image phenomenon7 image3)
	(have_image phenomenon8 image3)
	(have_image planet9 thermograph0)
	(have_image star10 spectrograph1)
))

)
