(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	infrared0 - mode
	spectrograph1 - mode
	infrared2 - mode
	thermograph3 - mode
	groundstation2 - direction
	star3 - direction
	groundstation0 - direction
	groundstation1 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared2)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
)
(:goal (and
	(pointing satellite2 groundstation1)
	(have_image phenomenon4 thermograph3)
	(have_image star5 thermograph3)
))

)
