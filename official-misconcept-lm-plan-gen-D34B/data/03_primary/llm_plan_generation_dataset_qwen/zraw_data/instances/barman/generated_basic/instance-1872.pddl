(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot323 shot493 - shot
      ingredient153 ingredient466 - ingredient
      cocktail108 - cocktail
      dispenser327 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot323)
  (ontable shot493)
  (dispenses dispenser327 ingredient153)
  (dispenses dispenser407 ingredient466)
  (clean shaker180)
  (clean shot323)
  (clean shot493)
  (empty shaker180)
  (empty shot323)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient466)
  (cocktail-part2 cocktail108 ingredient153)
)
 (:goal
  (and
      (contains shot323 cocktail108)
)))
