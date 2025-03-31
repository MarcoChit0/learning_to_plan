(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot297 - shot
      ingredient216 ingredient179 - ingredient
      cocktail444 - cocktail
      dispenser472 dispenser350 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot297)
  (dispenses dispenser472 ingredient216)
  (dispenses dispenser350 ingredient179)
  (clean shaker77)
  (clean shot297)
  (empty shaker77)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail444 ingredient179)
  (cocktail-part2 cocktail444 ingredient216)
)
 (:goal
  (and
      (contains shot297 cocktail444)
)))
