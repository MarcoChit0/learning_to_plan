(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot179 shot243 - shot
      ingredient130 ingredient329 ingredient283 - ingredient
      cocktail1 - cocktail
      dispenser119 dispenser265 dispenser494 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot179)
  (ontable shot243)
  (dispenses dispenser119 ingredient130)
  (dispenses dispenser265 ingredient329)
  (dispenses dispenser494 ingredient283)
  (clean shaker260)
  (clean shot179)
  (clean shot243)
  (empty shaker260)
  (empty shot179)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient130)
  (cocktail-part2 cocktail1 ingredient329)
)
 (:goal
  (and
      (contains shot179 cocktail1)
)))
