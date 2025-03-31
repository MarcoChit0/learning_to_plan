(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot59 - shot
      ingredient471 ingredient12 - ingredient
      cocktail1 - cocktail
      dispenser470 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot59)
  (dispenses dispenser470 ingredient471)
  (dispenses dispenser224 ingredient12)
  (clean shaker132)
  (clean shot59)
  (empty shaker132)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient12)
  (cocktail-part2 cocktail1 ingredient471)
)
 (:goal
  (and
      (contains shot59 cocktail1)
)))
