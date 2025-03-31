(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot445 shot207 shot270 - shot
      ingredient412 ingredient55 - ingredient
      cocktail150 - cocktail
      dispenser324 dispenser218 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot445)
  (ontable shot207)
  (ontable shot270)
  (dispenses dispenser324 ingredient412)
  (dispenses dispenser218 ingredient55)
  (clean shaker151)
  (clean shot445)
  (clean shot207)
  (clean shot270)
  (empty shaker151)
  (empty shot445)
  (empty shot207)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail150 ingredient55)
  (cocktail-part2 cocktail150 ingredient412)
)
 (:goal
  (and
      (contains shot445 cocktail150)
      (contains shot207 cocktail150)
)))
