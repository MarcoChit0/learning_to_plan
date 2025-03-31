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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	infrared3 - mode
	image2 - mode
	groundstation4 - direction
	star2 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation3 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	phenomenon9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 image2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon9)
	(supports instrument7 spectrograph1)
	(supports instrument7 image2)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared3)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star8)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation3)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(pointing satellite0 planet7)
	(pointing satellite1 groundstation0)
	(have_image star5 infrared3)
	(have_image phenomenon6 thermograph0)
	(have_image planet7 infrared3)
	(have_image star8 thermograph0)
	(have_image phenomenon9 thermograph0)
	(have_image planet10 image2)
))

)
