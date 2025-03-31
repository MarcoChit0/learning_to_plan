(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image0 - mode
	spectrograph1 - mode
	groundstation3 - direction
	groundstation2 - direction
	groundstation0 - direction
	groundstation4 - direction
	star1 - direction
	star5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(have_image star5 image0)
))

)
