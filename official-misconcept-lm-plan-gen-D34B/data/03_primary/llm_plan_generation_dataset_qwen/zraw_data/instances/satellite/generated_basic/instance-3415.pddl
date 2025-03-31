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
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image1 - mode
	infrared0 - mode
	spectrograph2 - mode
	groundstation2 - direction
	groundstation3 - direction
	star1 - direction
	groundstation0 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument6 infrared0)
	(supports instrument6 image1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite3 groundstation2)
	(have_image planet4 infrared0)
))

)
