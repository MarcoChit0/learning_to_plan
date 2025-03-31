(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot497 shot406 - shot
      ingredient329 ingredient439 ingredient229 ingredient399 - ingredient
      cocktail1 - cocktail
      dispenser47 dispenser434 dispenser336 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot497)
  (ontable shot406)
  (dispenses dispenser47 ingredient329)
  (dispenses dispenser434 ingredient439)
  (dispenses dispenser336 ingredient229)
  (dispenses dispenser285 ingredient399)
  (clean shaker490)
  (clean shot497)
  (clean shot406)
  (empty shaker490)
  (empty shot497)
  (empty shot406)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient229)
  (cocktail-part2 cocktail1 ingredient329)
)
 (:goal
  (and
      (contains shot497 cocktail1)
)))
