(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot4 - shot
      ingredient49 ingredient53 ingredient22 - ingredient
      cocktail1 - cocktail
      dispenser490 dispenser101 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot4)
  (dispenses dispenser490 ingredient49)
  (dispenses dispenser101 ingredient53)
  (dispenses dispenser191 ingredient22)
  (clean shaker468)
  (clean shot4)
  (empty shaker468)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient53)
  (cocktail-part2 cocktail1 ingredient22)
)
 (:goal
  (and
      (contains shot4 cocktail1)
)))
