(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot204 - shot
      ingredient445 ingredient74 ingredient18 ingredient389 - ingredient
      cocktail20 - cocktail
      dispenser411 dispenser438 dispenser397 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot204)
  (dispenses dispenser411 ingredient445)
  (dispenses dispenser438 ingredient74)
  (dispenses dispenser397 ingredient18)
  (dispenses dispenser100 ingredient389)
  (clean shaker247)
  (clean shot204)
  (empty shaker247)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail20 ingredient389)
  (cocktail-part2 cocktail20 ingredient18)
)
 (:goal
  (and
      (contains shot204 cocktail20)
)))
