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
	infrared1 - mode
	spectrograph2 - mode
	image0 - mode
	star4 - direction
	star3 - direction
	groundstation0 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 image0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 star3)
	(have_image phenomenon6 image0)
))

)
