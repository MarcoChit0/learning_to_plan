(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot31 shot461 shot206 - shot
      ingredient250 ingredient445 - ingredient
      cocktail189 - cocktail
      dispenser395 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot31)
  (ontable shot461)
  (ontable shot206)
  (dispenses dispenser395 ingredient250)
  (dispenses dispenser470 ingredient445)
  (clean shaker408)
  (clean shot31)
  (clean shot461)
  (clean shot206)
  (empty shaker408)
  (empty shot31)
  (empty shot461)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail189 ingredient250)
  (cocktail-part2 cocktail189 ingredient445)
)
 (:goal
  (and
      (contains shot31 cocktail189)
      (contains shot461 cocktail189)
)))
