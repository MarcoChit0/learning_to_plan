(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot445 - shot
      ingredient90 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser182 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot445)
  (dispenses dispenser182 ingredient90)
  (dispenses dispenser272 ingredient466)
  (clean shaker244)
  (clean shot445)
  (empty shaker244)
  (empty shot445)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient466)
  (cocktail-part2 cocktail1 ingredient90)
)
 (:goal
  (and
      (contains shot445 cocktail1)
)))
