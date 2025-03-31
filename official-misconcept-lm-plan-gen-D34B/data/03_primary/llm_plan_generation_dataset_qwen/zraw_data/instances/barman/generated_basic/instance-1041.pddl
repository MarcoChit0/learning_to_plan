(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot326 shot10 shot59 - shot
      ingredient8 ingredient328 ingredient445 ingredient429 - ingredient
      cocktail1 - cocktail
      dispenser170 dispenser51 dispenser332 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot326)
  (ontable shot10)
  (ontable shot59)
  (dispenses dispenser170 ingredient8)
  (dispenses dispenser51 ingredient328)
  (dispenses dispenser332 ingredient445)
  (dispenses dispenser59 ingredient429)
  (clean shaker22)
  (clean shot326)
  (clean shot10)
  (clean shot59)
  (empty shaker22)
  (empty shot326)
  (empty shot10)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient328)
  (cocktail-part2 cocktail1 ingredient429)
)
 (:goal
  (and
      (contains shot326 cocktail1)
      (contains shot10 cocktail1)
)))
