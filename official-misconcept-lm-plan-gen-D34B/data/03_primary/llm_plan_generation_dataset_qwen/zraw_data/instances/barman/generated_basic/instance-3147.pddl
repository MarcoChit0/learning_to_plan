(define (problem prob)
 (:domain barman)
 (:objects 
      shaker127 - shaker
      left right - hand
      shot328 shot329 shot188 - shot
      ingredient432 ingredient328 ingredient227 ingredient453 - ingredient
      cocktail174 - cocktail
      dispenser451 dispenser267 dispenser97 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker127)
  (ontable shot328)
  (ontable shot329)
  (ontable shot188)
  (dispenses dispenser451 ingredient432)
  (dispenses dispenser267 ingredient328)
  (dispenses dispenser97 ingredient227)
  (dispenses dispenser254 ingredient453)
  (clean shaker127)
  (clean shot328)
  (clean shot329)
  (clean shot188)
  (empty shaker127)
  (empty shot328)
  (empty shot329)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker127 l0)
  (shaker-level shaker127 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient227)
  (cocktail-part2 cocktail174 ingredient453)
)
 (:goal
  (and
      (contains shot328 cocktail174)
      (contains shot329 cocktail174)
)))
