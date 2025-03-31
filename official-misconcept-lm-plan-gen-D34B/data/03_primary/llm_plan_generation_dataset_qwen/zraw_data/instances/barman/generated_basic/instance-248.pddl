(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot438 - shot
      ingredient298 ingredient376 - ingredient
      cocktail1 - cocktail
      dispenser286 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot438)
  (dispenses dispenser286 ingredient298)
  (dispenses dispenser257 ingredient376)
  (clean shaker26)
  (clean shot438)
  (empty shaker26)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient298)
  (cocktail-part2 cocktail1 ingredient376)
)
 (:goal
  (and
      (contains shot438 cocktail1)
)))
