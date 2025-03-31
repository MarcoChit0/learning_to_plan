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
	infrared2 - mode
	spectrograph3 - mode
	spectrograph0 - mode
	infrared1 - mode
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 spectrograph3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star5)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite0 star5)
	(have_image planet6 spectrograph3)
))

)
