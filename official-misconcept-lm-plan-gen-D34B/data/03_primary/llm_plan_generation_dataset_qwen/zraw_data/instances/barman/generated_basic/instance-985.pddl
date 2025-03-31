(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot426 shot340 - shot
      ingredient82 ingredient288 ingredient141 ingredient171 - ingredient
      cocktail1 - cocktail
      dispenser489 dispenser204 dispenser497 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot426)
  (ontable shot340)
  (dispenses dispenser489 ingredient82)
  (dispenses dispenser204 ingredient288)
  (dispenses dispenser497 ingredient141)
  (dispenses dispenser45 ingredient171)
  (clean shaker328)
  (clean shot426)
  (clean shot340)
  (empty shaker328)
  (empty shot426)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient171)
  (cocktail-part2 cocktail1 ingredient141)
)
 (:goal
  (and
      (contains shot426 cocktail1)
)))
