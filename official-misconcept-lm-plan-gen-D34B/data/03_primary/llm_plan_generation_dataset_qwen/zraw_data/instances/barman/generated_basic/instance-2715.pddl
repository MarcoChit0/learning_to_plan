(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot361 shot371 shot480 - shot
      ingredient125 ingredient312 ingredient107 - ingredient
      cocktail330 - cocktail
      dispenser280 dispenser245 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot361)
  (ontable shot371)
  (ontable shot480)
  (dispenses dispenser280 ingredient125)
  (dispenses dispenser245 ingredient312)
  (dispenses dispenser120 ingredient107)
  (clean shaker425)
  (clean shot361)
  (clean shot371)
  (clean shot480)
  (empty shaker425)
  (empty shot361)
  (empty shot371)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient312)
  (cocktail-part2 cocktail330 ingredient107)
)
 (:goal
  (and
      (contains shot361 cocktail330)
      (contains shot371 cocktail330)
)))
