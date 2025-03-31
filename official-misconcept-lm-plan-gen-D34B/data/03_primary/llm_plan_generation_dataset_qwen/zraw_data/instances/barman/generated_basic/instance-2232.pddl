(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot393 - shot
      ingredient28 ingredient39 - ingredient
      cocktail210 - cocktail
      dispenser200 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot393)
  (dispenses dispenser200 ingredient28)
  (dispenses dispenser93 ingredient39)
  (clean shaker355)
  (clean shot393)
  (empty shaker355)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient39)
  (cocktail-part2 cocktail210 ingredient28)
)
 (:goal
  (and
      (contains shot393 cocktail210)
)))
