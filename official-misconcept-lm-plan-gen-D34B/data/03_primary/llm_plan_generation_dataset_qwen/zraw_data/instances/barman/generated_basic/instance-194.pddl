(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot290 - shot
      ingredient479 ingredient485 ingredient114 - ingredient
      cocktail1 - cocktail
      dispenser466 dispenser41 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot290)
  (dispenses dispenser466 ingredient479)
  (dispenses dispenser41 ingredient485)
  (dispenses dispenser388 ingredient114)
  (clean shaker37)
  (clean shot290)
  (empty shaker37)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient485)
  (cocktail-part2 cocktail1 ingredient479)
)
 (:goal
  (and
      (contains shot290 cocktail1)
)))
