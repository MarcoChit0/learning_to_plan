(define (problem prob)
 (:domain barman)
 (:objects 
      shaker223 - shaker
      left right - hand
      shot96 shot244 shot134 - shot
      ingredient247 ingredient236 ingredient270 ingredient142 - ingredient
      cocktail1 - cocktail
      dispenser162 dispenser131 dispenser137 dispenser280 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker223)
  (ontable shot96)
  (ontable shot244)
  (ontable shot134)
  (dispenses dispenser162 ingredient247)
  (dispenses dispenser131 ingredient236)
  (dispenses dispenser137 ingredient270)
  (dispenses dispenser280 ingredient142)
  (clean shaker223)
  (clean shot96)
  (clean shot244)
  (clean shot134)
  (empty shaker223)
  (empty shot96)
  (empty shot244)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker223 l0)
  (shaker-level shaker223 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient236)
  (cocktail-part2 cocktail1 ingredient270)
)
 (:goal
  (and
      (contains shot96 cocktail1)
      (contains shot244 ingredient236)
)))
