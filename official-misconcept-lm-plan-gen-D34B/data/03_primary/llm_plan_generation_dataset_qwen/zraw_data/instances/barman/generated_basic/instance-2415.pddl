(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot134 - shot
      ingredient304 ingredient212 - ingredient
      cocktail447 - cocktail
      dispenser66 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot134)
  (dispenses dispenser66 ingredient304)
  (dispenses dispenser122 ingredient212)
  (clean shaker109)
  (clean shot134)
  (empty shaker109)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail447 ingredient212)
  (cocktail-part2 cocktail447 ingredient304)
)
 (:goal
  (and
      (contains shot134 cocktail447)
)))
