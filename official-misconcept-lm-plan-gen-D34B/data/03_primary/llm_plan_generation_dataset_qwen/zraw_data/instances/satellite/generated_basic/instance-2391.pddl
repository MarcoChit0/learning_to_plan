(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph3 - mode
	thermograph2 - mode
	image1 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph3)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image1)
	(supports instrument8 thermograph2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation4)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite2 groundstation4)
	(have_image star5 image1)
	(have_image phenomenon6 image1)
))

)
