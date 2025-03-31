(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	spectrograph0 - mode
	groundstation0 - direction
	star1 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation2 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 groundstation2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star5)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite3 groundstation0)
	(have_image star6 spectrograph0)
))

)
