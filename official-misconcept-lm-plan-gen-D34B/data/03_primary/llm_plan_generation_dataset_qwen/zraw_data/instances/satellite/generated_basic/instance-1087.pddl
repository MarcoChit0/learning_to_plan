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
	infrared1 - mode
	spectrograph0 - mode
	infrared2 - mode
	star0 - direction
	star4 - direction
	groundstation2 - direction
	star5 - direction
	star3 - direction
	star1 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star5)
	(supports instrument2 infrared2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite0 star6)
	(have_image star6 infrared1)
	(have_image phenomenon7 spectrograph0)
	(have_image planet8 infrared2)
))

)
