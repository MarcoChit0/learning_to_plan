(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot371 - shot
      ingredient241 ingredient190 - ingredient
      cocktail157 - cocktail
      dispenser327 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot371)
  (dispenses dispenser327 ingredient241)
  (dispenses dispenser376 ingredient190)
  (clean shaker345)
  (clean shot371)
  (empty shaker345)
  (empty shot371)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail157 ingredient190)
  (cocktail-part2 cocktail157 ingredient241)
)
 (:goal
  (and
      (contains shot371 cocktail157)
)))
