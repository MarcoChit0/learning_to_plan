(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot439 shot51 shot414 - shot
      ingredient96 ingredient492 - ingredient
      cocktail51 - cocktail
      dispenser396 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot439)
  (ontable shot51)
  (ontable shot414)
  (dispenses dispenser396 ingredient96)
  (dispenses dispenser193 ingredient492)
  (clean shaker390)
  (clean shot439)
  (clean shot51)
  (clean shot414)
  (empty shaker390)
  (empty shot439)
  (empty shot51)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail51 ingredient96)
  (cocktail-part2 cocktail51 ingredient492)
)
 (:goal
  (and
      (contains shot439 cocktail51)
      (contains shot51 cocktail51)
)))
