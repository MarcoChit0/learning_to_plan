(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot115 shot427 shot366 - shot
      ingredient258 ingredient260 ingredient392 - ingredient
      cocktail325 - cocktail
      dispenser342 dispenser368 dispenser253 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot115)
  (ontable shot427)
  (ontable shot366)
  (dispenses dispenser342 ingredient258)
  (dispenses dispenser368 ingredient260)
  (dispenses dispenser253 ingredient392)
  (clean shaker445)
  (clean shot115)
  (clean shot427)
  (clean shot366)
  (empty shaker445)
  (empty shot115)
  (empty shot427)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail325 ingredient260)
  (cocktail-part2 cocktail325 ingredient258)
)
 (:goal
  (and
      (contains shot115 cocktail325)
      (contains shot427 ingredient258)
)))
