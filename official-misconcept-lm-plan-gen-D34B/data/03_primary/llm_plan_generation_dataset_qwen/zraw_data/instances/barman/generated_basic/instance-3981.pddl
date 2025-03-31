(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot376 shot47 - shot
      ingredient424 ingredient380 ingredient81 ingredient8 - ingredient
      cocktail98 - cocktail
      dispenser7 dispenser206 dispenser84 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot376)
  (ontable shot47)
  (dispenses dispenser7 ingredient424)
  (dispenses dispenser206 ingredient380)
  (dispenses dispenser84 ingredient81)
  (dispenses dispenser480 ingredient8)
  (clean shaker445)
  (clean shot376)
  (clean shot47)
  (empty shaker445)
  (empty shot376)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail98 ingredient380)
  (cocktail-part2 cocktail98 ingredient81)
)
 (:goal
  (and
      (contains shot376 cocktail98)
)))
