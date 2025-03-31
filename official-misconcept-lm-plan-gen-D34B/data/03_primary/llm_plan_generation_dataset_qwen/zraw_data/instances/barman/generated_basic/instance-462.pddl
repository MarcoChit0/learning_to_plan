(define (problem prob)
 (:domain barman)
 (:objects 
      shaker100 - shaker
      left right - hand
      shot188 shot445 shot37 - shot
      ingredient385 ingredient17 ingredient170 ingredient347 - ingredient
      cocktail1 - cocktail
      dispenser355 dispenser108 dispenser487 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker100)
  (ontable shot188)
  (ontable shot445)
  (ontable shot37)
  (dispenses dispenser355 ingredient385)
  (dispenses dispenser108 ingredient17)
  (dispenses dispenser487 ingredient170)
  (dispenses dispenser456 ingredient347)
  (clean shaker100)
  (clean shot188)
  (clean shot445)
  (clean shot37)
  (empty shaker100)
  (empty shot188)
  (empty shot445)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker100 l0)
  (shaker-level shaker100 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient347)
  (cocktail-part2 cocktail1 ingredient170)
)
 (:goal
  (and
      (contains shot188 cocktail1)
      (contains shot445 ingredient385)
)))
