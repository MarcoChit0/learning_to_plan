(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared2 - mode
	infrared1 - mode
	infrared3 - mode
	spectrograph0 - mode
	groundstation2 - direction
	groundstation0 - direction
	star1 - direction
	planet3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 infrared2)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared2)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared1)
	(supports instrument7 infrared3)
	(supports instrument7 infrared2)
	(calibration_target instrument7 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite2 planet4)
	(have_image planet3 infrared3)
	(have_image planet4 infrared2)
))

)
