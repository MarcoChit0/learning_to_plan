(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot494 - shot
      ingredient203 ingredient286 ingredient49 - ingredient
      cocktail364 - cocktail
      dispenser240 dispenser372 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot494)
  (dispenses dispenser240 ingredient203)
  (dispenses dispenser372 ingredient286)
  (dispenses dispenser259 ingredient49)
  (clean shaker474)
  (clean shot494)
  (empty shaker474)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail364 ingredient286)
  (cocktail-part2 cocktail364 ingredient203)
)
 (:goal
  (and
      (contains shot494 cocktail364)
)))
