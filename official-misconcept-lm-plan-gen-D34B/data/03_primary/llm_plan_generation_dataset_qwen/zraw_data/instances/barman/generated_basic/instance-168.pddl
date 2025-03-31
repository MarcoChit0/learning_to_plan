(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot326 shot107 - shot
      ingredient384 ingredient64 ingredient344 - ingredient
      cocktail1 - cocktail
      dispenser260 dispenser426 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot326)
  (ontable shot107)
  (dispenses dispenser260 ingredient384)
  (dispenses dispenser426 ingredient64)
  (dispenses dispenser69 ingredient344)
  (clean shaker445)
  (clean shot326)
  (clean shot107)
  (empty shaker445)
  (empty shot326)
  (empty shot107)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient384)
  (cocktail-part2 cocktail1 ingredient64)
)
 (:goal
  (and
      (contains shot326 cocktail1)
)))
