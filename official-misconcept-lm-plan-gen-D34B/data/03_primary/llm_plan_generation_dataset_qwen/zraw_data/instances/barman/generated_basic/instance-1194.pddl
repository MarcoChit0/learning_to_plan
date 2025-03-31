(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot445 shot324 - shot
      ingredient390 ingredient424 - ingredient
      cocktail177 - cocktail
      dispenser45 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot445)
  (ontable shot324)
  (dispenses dispenser45 ingredient390)
  (dispenses dispenser146 ingredient424)
  (clean shaker234)
  (clean shot445)
  (clean shot324)
  (empty shaker234)
  (empty shot445)
  (empty shot324)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient424)
  (cocktail-part2 cocktail177 ingredient390)
)
 (:goal
  (and
      (contains shot445 cocktail177)
)))
