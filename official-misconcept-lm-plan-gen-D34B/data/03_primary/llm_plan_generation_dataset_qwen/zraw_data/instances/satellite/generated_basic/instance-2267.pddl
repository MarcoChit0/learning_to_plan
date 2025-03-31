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
	thermograph0 - mode
	image1 - mode
	infrared3 - mode
	spectrograph2 - mode
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet3)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite3 star0)
	(have_image phenomenon1 thermograph0)
	(have_image phenomenon2 infrared3)
	(have_image planet3 image1)
))

)
