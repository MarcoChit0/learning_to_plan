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
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	infrared3 - mode
	spectrograph2 - mode
	infrared1 - mode
	infrared0 - mode
	groundstation2 - direction
	star3 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument2 infrared3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument4 infrared3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 infrared1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation4)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(have_image planet5 infrared3)
	(have_image phenomenon6 infrared3)
))

)
