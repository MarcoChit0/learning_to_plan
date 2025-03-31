(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot293 shot5 - shot
      ingredient472 ingredient397 ingredient238 ingredient103 - ingredient
      cocktail1 - cocktail
      dispenser1 dispenser125 dispenser136 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot293)
  (ontable shot5)
  (dispenses dispenser1 ingredient472)
  (dispenses dispenser125 ingredient397)
  (dispenses dispenser136 ingredient238)
  (dispenses dispenser385 ingredient103)
  (clean shaker110)
  (clean shot293)
  (clean shot5)
  (empty shaker110)
  (empty shot293)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient472)
  (cocktail-part2 cocktail1 ingredient103)
)
 (:goal
  (and
      (contains shot293 cocktail1)
)))
