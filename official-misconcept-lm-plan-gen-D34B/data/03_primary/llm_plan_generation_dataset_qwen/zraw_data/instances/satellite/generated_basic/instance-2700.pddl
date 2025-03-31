(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	infrared2 - mode
	spectrograph0 - mode
	thermograph3 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star0 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image phenomenon4 spectrograph0)
))

)
