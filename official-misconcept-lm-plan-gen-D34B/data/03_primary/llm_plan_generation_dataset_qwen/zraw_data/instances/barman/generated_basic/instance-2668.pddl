(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot337 - shot
      ingredient137 ingredient294 ingredient247 ingredient162 - ingredient
      cocktail119 - cocktail
      dispenser55 dispenser22 dispenser419 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot337)
  (dispenses dispenser55 ingredient137)
  (dispenses dispenser22 ingredient294)
  (dispenses dispenser419 ingredient247)
  (dispenses dispenser426 ingredient162)
  (clean shaker345)
  (clean shot337)
  (empty shaker345)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient137)
  (cocktail-part2 cocktail119 ingredient294)
)
 (:goal
  (and
      (contains shot337 cocktail119)
)))
