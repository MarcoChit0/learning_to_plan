(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot415 - shot
      ingredient111 ingredient102 - ingredient
      cocktail1 - cocktail
      dispenser182 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot415)
  (dispenses dispenser182 ingredient111)
  (dispenses dispenser195 ingredient102)
  (clean shaker270)
  (clean shot415)
  (empty shaker270)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient102)
  (cocktail-part2 cocktail1 ingredient111)
)
 (:goal
  (and
      (contains shot415 cocktail1)
)))
