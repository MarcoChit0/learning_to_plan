(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot445 shot288 shot311 - shot
      ingredient311 ingredient302 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser175 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot445)
  (ontable shot288)
  (ontable shot311)
  (dispenses dispenser175 ingredient311)
  (dispenses dispenser354 ingredient302)
  (clean shaker449)
  (clean shot445)
  (clean shot288)
  (clean shot311)
  (empty shaker449)
  (empty shot445)
  (empty shot288)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient302)
  (cocktail-part2 cocktail1 ingredient311)
  (cocktail-part1 cocktail2 ingredient302)
  (cocktail-part2 cocktail2 ingredient311)
)
 (:goal
  (and
      (contains shot445 cocktail2)
      (contains shot288 cocktail1)
)))
