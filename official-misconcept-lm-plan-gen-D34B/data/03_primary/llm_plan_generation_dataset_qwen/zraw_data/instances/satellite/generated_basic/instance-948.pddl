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
	infrared1 - mode
	infrared2 - mode
	spectrograph3 - mode
	image0 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 image0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph3)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star6)
	(have_image phenomenon4 spectrograph3)
	(have_image star5 spectrograph3)
	(have_image star6 image0)
))

)
