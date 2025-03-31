(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot74 shot376 - shot
      ingredient152 ingredient350 ingredient63 ingredient256 - ingredient
      cocktail1 - cocktail
      dispenser359 dispenser328 dispenser21 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot74)
  (ontable shot376)
  (dispenses dispenser359 ingredient152)
  (dispenses dispenser328 ingredient350)
  (dispenses dispenser21 ingredient63)
  (dispenses dispenser267 ingredient256)
  (clean shaker45)
  (clean shot74)
  (clean shot376)
  (empty shaker45)
  (empty shot74)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient63)
  (cocktail-part2 cocktail1 ingredient350)
)
 (:goal
  (and
      (contains shot74 cocktail1)
)))
