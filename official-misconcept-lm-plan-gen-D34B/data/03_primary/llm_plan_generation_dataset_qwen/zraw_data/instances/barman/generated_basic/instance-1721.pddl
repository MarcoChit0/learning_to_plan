(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot5 shot66 - shot
      ingredient188 ingredient421 ingredient160 ingredient4 - ingredient
      cocktail494 - cocktail
      dispenser186 dispenser124 dispenser238 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot5)
  (ontable shot66)
  (dispenses dispenser186 ingredient188)
  (dispenses dispenser124 ingredient421)
  (dispenses dispenser238 ingredient160)
  (dispenses dispenser207 ingredient4)
  (clean shaker428)
  (clean shot5)
  (clean shot66)
  (empty shaker428)
  (empty shot5)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail494 ingredient421)
  (cocktail-part2 cocktail494 ingredient4)
)
 (:goal
  (and
      (contains shot5 cocktail494)
)))
