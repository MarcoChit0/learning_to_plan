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
	infrared1 - mode
	image0 - mode
	spectrograph2 - mode
	groundstation1 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 infrared1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star7)
	(pointing satellite1 planet6)
	(pointing satellite2 planet8)
	(pointing satellite3 star4)
	(have_image star3 image0)
	(have_image star4 infrared1)
	(have_image phenomenon5 infrared1)
	(have_image planet6 spectrograph2)
	(have_image star7 infrared1)
	(have_image planet8 image0)
))

)
