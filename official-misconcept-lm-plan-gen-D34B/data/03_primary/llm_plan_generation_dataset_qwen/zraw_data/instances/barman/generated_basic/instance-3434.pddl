(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot189 shot299 - shot
      ingredient496 ingredient292 ingredient189 - ingredient
      cocktail168 - cocktail
      dispenser25 dispenser102 dispenser408 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot189)
  (ontable shot299)
  (dispenses dispenser25 ingredient496)
  (dispenses dispenser102 ingredient292)
  (dispenses dispenser408 ingredient189)
  (clean shaker461)
  (clean shot189)
  (clean shot299)
  (empty shaker461)
  (empty shot189)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail168 ingredient292)
  (cocktail-part2 cocktail168 ingredient189)
)
 (:goal
  (and
      (contains shot189 cocktail168)
)))
