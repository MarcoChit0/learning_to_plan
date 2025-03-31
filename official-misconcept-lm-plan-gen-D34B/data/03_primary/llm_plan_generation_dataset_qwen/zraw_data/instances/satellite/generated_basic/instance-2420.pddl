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
	spectrograph3 - mode
	infrared2 - mode
	thermograph0 - mode
	thermograph1 - mode
	groundstation2 - direction
	groundstation4 - direction
	star1 - direction
	star3 - direction
	groundstation0 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 spectrograph3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(have_image star5 thermograph0)
	(have_image star6 infrared2)
))

)
