(define (problem prob)
 (:domain barman)
 (:objects 
      shaker430 - shaker
      left right - hand
      shot350 shot425 - shot
      ingredient318 ingredient104 ingredient118 - ingredient
      cocktail328 - cocktail
      dispenser439 dispenser125 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker430)
  (ontable shot350)
  (ontable shot425)
  (dispenses dispenser439 ingredient318)
  (dispenses dispenser125 ingredient104)
  (dispenses dispenser49 ingredient118)
  (clean shaker430)
  (clean shot350)
  (clean shot425)
  (empty shaker430)
  (empty shot350)
  (empty shot425)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker430 l0)
  (shaker-level shaker430 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient104)
  (cocktail-part2 cocktail328 ingredient318)
)
 (:goal
  (and
      (contains shot350 cocktail328)
)))
