(define (problem prob)
 (:domain barman)
 (:objects 
      shaker193 - shaker
      left right - hand
      shot263 - shot
      ingredient454 ingredient409 - ingredient
      cocktail413 - cocktail
      dispenser200 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker193)
  (ontable shot263)
  (dispenses dispenser200 ingredient454)
  (dispenses dispenser369 ingredient409)
  (clean shaker193)
  (clean shot263)
  (empty shaker193)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker193 l0)
  (shaker-level shaker193 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail413 ingredient454)
  (cocktail-part2 cocktail413 ingredient409)
)
 (:goal
  (and
      (contains shot263 cocktail413)
)))
