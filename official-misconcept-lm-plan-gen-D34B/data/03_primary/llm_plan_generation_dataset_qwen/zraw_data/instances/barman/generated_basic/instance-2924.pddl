(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot38 - shot
      ingredient241 ingredient224 - ingredient
      cocktail439 - cocktail
      dispenser325 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot38)
  (dispenses dispenser325 ingredient241)
  (dispenses dispenser66 ingredient224)
  (clean shaker51)
  (clean shot38)
  (empty shaker51)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient241)
  (cocktail-part2 cocktail439 ingredient224)
)
 (:goal
  (and
      (contains shot38 cocktail439)
)))
