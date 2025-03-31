(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image3 - mode
	thermograph0 - mode
	spectrograph2 - mode
	image1 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image3)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(supports instrument5 image3)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(supports instrument6 image3)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite2 star1)
	(have_image star1 image3)
	(have_image phenomenon2 spectrograph2)
	(have_image phenomenon3 spectrograph2)
))

)
