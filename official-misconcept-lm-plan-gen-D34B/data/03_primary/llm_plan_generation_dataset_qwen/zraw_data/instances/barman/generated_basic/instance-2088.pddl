(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot231 shot280 - shot
      ingredient12 ingredient298 - ingredient
      cocktail427 - cocktail
      dispenser357 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot231)
  (ontable shot280)
  (dispenses dispenser357 ingredient12)
  (dispenses dispenser146 ingredient298)
  (clean shaker425)
  (clean shot231)
  (clean shot280)
  (empty shaker425)
  (empty shot231)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail427 ingredient298)
  (cocktail-part2 cocktail427 ingredient12)
)
 (:goal
  (and
      (contains shot231 cocktail427)
)))
