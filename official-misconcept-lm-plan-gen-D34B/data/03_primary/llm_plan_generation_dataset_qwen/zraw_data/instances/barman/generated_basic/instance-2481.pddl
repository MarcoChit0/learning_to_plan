(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot467 - shot
      ingredient127 ingredient445 ingredient345 - ingredient
      cocktail207 - cocktail
      dispenser56 dispenser280 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot467)
  (dispenses dispenser56 ingredient127)
  (dispenses dispenser280 ingredient445)
  (dispenses dispenser20 ingredient345)
  (clean shaker374)
  (clean shot467)
  (empty shaker374)
  (empty shot467)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient127)
  (cocktail-part2 cocktail207 ingredient345)
)
 (:goal
  (and
      (contains shot467 cocktail207)
)))
