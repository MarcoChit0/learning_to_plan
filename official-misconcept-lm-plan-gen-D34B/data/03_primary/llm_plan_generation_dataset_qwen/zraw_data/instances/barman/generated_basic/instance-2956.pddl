(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot131 - shot
      ingredient446 ingredient238 ingredient461 ingredient212 - ingredient
      cocktail170 - cocktail
      dispenser254 dispenser105 dispenser247 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot131)
  (dispenses dispenser254 ingredient446)
  (dispenses dispenser105 ingredient238)
  (dispenses dispenser247 ingredient461)
  (dispenses dispenser382 ingredient212)
  (clean shaker273)
  (clean shot131)
  (empty shaker273)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient461)
  (cocktail-part2 cocktail170 ingredient212)
)
 (:goal
  (and
      (contains shot131 cocktail170)
)))
