(define (problem prob)
 (:domain barman)
 (:objects 
      shaker173 - shaker
      left right - hand
      shot445 shot193 - shot
      ingredient450 ingredient246 - ingredient
      cocktail466 - cocktail
      dispenser443 dispenser117 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker173)
  (ontable shot445)
  (ontable shot193)
  (dispenses dispenser443 ingredient450)
  (dispenses dispenser117 ingredient246)
  (clean shaker173)
  (clean shot445)
  (clean shot193)
  (empty shaker173)
  (empty shot445)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker173 l0)
  (shaker-level shaker173 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail466 ingredient246)
  (cocktail-part2 cocktail466 ingredient450)
)
 (:goal
  (and
      (contains shot445 cocktail466)
)))
