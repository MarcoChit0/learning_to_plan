(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot22 - shot
      ingredient444 ingredient134 - ingredient
      cocktail1 - cocktail
      dispenser55 dispenser451 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot22)
  (dispenses dispenser55 ingredient444)
  (dispenses dispenser451 ingredient134)
  (clean shaker488)
  (clean shot22)
  (empty shaker488)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient134)
  (cocktail-part2 cocktail1 ingredient444)
)
 (:goal
  (and
      (contains shot22 cocktail1)
)))
