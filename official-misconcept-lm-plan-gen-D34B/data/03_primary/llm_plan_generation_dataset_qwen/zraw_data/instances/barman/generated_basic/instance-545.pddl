(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot120 - shot
      ingredient107 ingredient39 ingredient240 - ingredient
      cocktail1 - cocktail
      dispenser41 dispenser102 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot120)
  (dispenses dispenser41 ingredient107)
  (dispenses dispenser102 ingredient39)
  (dispenses dispenser477 ingredient240)
  (clean shaker204)
  (clean shot120)
  (empty shaker204)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient107)
  (cocktail-part2 cocktail1 ingredient39)
)
 (:goal
  (and
      (contains shot120 cocktail1)
)))
