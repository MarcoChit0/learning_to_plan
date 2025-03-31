(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot428 shot442 - shot
      ingredient254 ingredient481 - ingredient
      cocktail171 - cocktail
      dispenser58 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot428)
  (ontable shot442)
  (dispenses dispenser58 ingredient254)
  (dispenses dispenser124 ingredient481)
  (clean shaker469)
  (clean shot428)
  (clean shot442)
  (empty shaker469)
  (empty shot428)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient481)
  (cocktail-part2 cocktail171 ingredient254)
)
 (:goal
  (and
      (contains shot428 cocktail171)
)))
