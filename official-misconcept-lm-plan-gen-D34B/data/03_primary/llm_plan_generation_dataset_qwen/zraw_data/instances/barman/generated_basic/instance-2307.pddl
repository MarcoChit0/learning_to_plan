(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot374 - shot
      ingredient45 ingredient38 - ingredient
      cocktail110 - cocktail
      dispenser90 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot374)
  (dispenses dispenser90 ingredient45)
  (dispenses dispenser272 ingredient38)
  (clean shaker474)
  (clean shot374)
  (empty shaker474)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient45)
  (cocktail-part2 cocktail110 ingredient38)
)
 (:goal
  (and
      (contains shot374 cocktail110)
)))
