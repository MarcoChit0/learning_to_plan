(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot328 shot484 - shot
      ingredient227 ingredient30 - ingredient
      cocktail477 - cocktail
      dispenser26 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot328)
  (ontable shot484)
  (dispenses dispenser26 ingredient227)
  (dispenses dispenser313 ingredient30)
  (clean shaker409)
  (clean shot328)
  (clean shot484)
  (empty shaker409)
  (empty shot328)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail477 ingredient227)
  (cocktail-part2 cocktail477 ingredient30)
)
 (:goal
  (and
      (contains shot328 cocktail477)
)))
