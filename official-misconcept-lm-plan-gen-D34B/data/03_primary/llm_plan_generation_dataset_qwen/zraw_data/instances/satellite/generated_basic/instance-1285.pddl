(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared2 - mode
	infrared1 - mode
	spectrograph0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation5 - direction
	groundstation0 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 spectrograph0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(have_image phenomenon6 infrared2)
	(have_image star7 infrared2)
	(have_image planet8 infrared1)
	(have_image phenomenon9 spectrograph0)
))

)
