(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot282 - shot
      ingredient423 ingredient156 ingredient298 - ingredient
      cocktail185 - cocktail
      dispenser453 dispenser102 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot282)
  (dispenses dispenser453 ingredient423)
  (dispenses dispenser102 ingredient156)
  (dispenses dispenser474 ingredient298)
  (clean shaker487)
  (clean shot282)
  (empty shaker487)
  (empty shot282)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail185 ingredient423)
  (cocktail-part2 cocktail185 ingredient298)
)
 (:goal
  (and
      (contains shot282 cocktail185)
)))
