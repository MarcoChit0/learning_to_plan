(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot294 shot195 shot431 - shot
      ingredient492 ingredient257 - ingredient
      cocktail1 - cocktail
      dispenser328 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot294)
  (ontable shot195)
  (ontable shot431)
  (dispenses dispenser328 ingredient492)
  (dispenses dispenser194 ingredient257)
  (clean shaker491)
  (clean shot294)
  (clean shot195)
  (clean shot431)
  (empty shaker491)
  (empty shot294)
  (empty shot195)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient492)
  (cocktail-part2 cocktail1 ingredient257)
)
 (:goal
  (and
      (contains shot294 cocktail1)
      (contains shot195 ingredient257)
)))
