(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot360 shot200 - shot
      ingredient428 ingredient49 ingredient304 - ingredient
      cocktail378 - cocktail
      dispenser457 dispenser122 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot360)
  (ontable shot200)
  (dispenses dispenser457 ingredient428)
  (dispenses dispenser122 ingredient49)
  (dispenses dispenser490 ingredient304)
  (clean shaker87)
  (clean shot360)
  (clean shot200)
  (empty shaker87)
  (empty shot360)
  (empty shot200)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail378 ingredient49)
  (cocktail-part2 cocktail378 ingredient304)
)
 (:goal
  (and
      (contains shot360 cocktail378)
)))
