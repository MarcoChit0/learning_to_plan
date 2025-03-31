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
	satellite3 - satellite
	instrument7 - instrument
	infrared3 - mode
	spectrograph1 - mode
	thermograph0 - mode
	infrared2 - mode
	star4 - direction
	star2 - direction
	groundstation1 - direction
	star0 - direction
	groundstation3 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 infrared2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared3)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument3 infrared3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument7 infrared2)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(have_image planet5 thermograph0)
	(have_image phenomenon6 infrared3)
))

)
