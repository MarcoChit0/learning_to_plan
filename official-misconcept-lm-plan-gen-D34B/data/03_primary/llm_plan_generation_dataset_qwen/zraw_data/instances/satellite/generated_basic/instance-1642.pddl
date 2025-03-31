(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared2 - mode
	spectrograph0 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument5 infrared2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 star4)
	(have_image star2 infrared1)
	(have_image planet3 infrared2)
	(have_image star4 infrared1)
))

)
