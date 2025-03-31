(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot491 shot425 - shot
      ingredient23 ingredient88 ingredient363 - ingredient
      cocktail1 - cocktail
      dispenser367 dispenser491 dispenser386 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot491)
  (ontable shot425)
  (dispenses dispenser367 ingredient23)
  (dispenses dispenser491 ingredient88)
  (dispenses dispenser386 ingredient363)
  (clean shaker45)
  (clean shot491)
  (clean shot425)
  (empty shaker45)
  (empty shot491)
  (empty shot425)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient23)
  (cocktail-part2 cocktail1 ingredient363)
)
 (:goal
  (and
      (contains shot491 cocktail1)
)))
