(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot426 - shot
      ingredient181 ingredient316 - ingredient
      cocktail1 - cocktail
      dispenser286 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot426)
  (dispenses dispenser286 ingredient181)
  (dispenses dispenser157 ingredient316)
  (clean shaker461)
  (clean shot426)
  (empty shaker461)
  (empty shot426)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient316)
  (cocktail-part2 cocktail1 ingredient181)
)
 (:goal
  (and
      (contains shot426 cocktail1)
)))
