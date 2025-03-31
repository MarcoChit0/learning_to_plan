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
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph0 - mode
	spectrograph3 - mode
	infrared1 - mode
	infrared2 - mode
	groundstation3 - direction
	star1 - direction
	star0 - direction
	star4 - direction
	groundstation2 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
	star9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph3)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared2)
	(supports instrument6 spectrograph3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image star5 thermograph0)
	(have_image phenomenon6 infrared2)
	(have_image planet7 spectrograph3)
	(have_image planet8 thermograph0)
	(have_image star9 thermograph0)
	(have_image planet10 thermograph0)
))

)
