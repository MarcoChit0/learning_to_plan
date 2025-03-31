(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	infrared3 - mode
	infrared2 - mode
	infrared1 - mode
	spectrograph0 - mode
	star0 - direction
	star3 - direction
	groundstation2 - direction
	groundstation1 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared2)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 infrared2)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 infrared3)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star9)
)
(:goal (and
	(pointing satellite0 star9)
	(pointing satellite1 planet5)
	(pointing satellite3 star3)
	(have_image phenomenon4 spectrograph0)
	(have_image planet5 infrared3)
	(have_image star6 spectrograph0)
	(have_image star7 infrared2)
	(have_image planet8 spectrograph0)
	(have_image star9 infrared2)
))

)
