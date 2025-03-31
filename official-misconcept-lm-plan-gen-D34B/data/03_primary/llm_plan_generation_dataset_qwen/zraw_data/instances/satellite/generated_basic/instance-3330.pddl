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
	infrared1 - mode
	infrared2 - mode
	spectrograph0 - mode
	groundstation2 - direction
	star1 - direction
	star0 - direction
	phenomenon3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument3 spectrograph0)
	(supports instrument3 infrared1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument5 infrared2)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 planet4)
	(have_image phenomenon3 infrared2)
	(have_image planet4 infrared2)
))

)
