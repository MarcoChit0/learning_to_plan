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
	thermograph2 - mode
	infrared0 - mode
	spectrograph3 - mode
	image1 - mode
	star1 - direction
	star0 - direction
	phenomenon2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 image1)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(have_image phenomenon2 thermograph2)
	(have_image star3 image1)
))

)
