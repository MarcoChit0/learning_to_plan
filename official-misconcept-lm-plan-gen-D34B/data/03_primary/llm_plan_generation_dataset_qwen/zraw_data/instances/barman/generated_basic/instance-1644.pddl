(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot308 shot247 shot499 - shot
      ingredient278 ingredient42 ingredient237 ingredient418 - ingredient
      cocktail294 - cocktail
      dispenser216 dispenser159 dispenser26 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot308)
  (ontable shot247)
  (ontable shot499)
  (dispenses dispenser216 ingredient278)
  (dispenses dispenser159 ingredient42)
  (dispenses dispenser26 ingredient237)
  (dispenses dispenser100 ingredient418)
  (clean shaker445)
  (clean shot308)
  (clean shot247)
  (clean shot499)
  (empty shaker445)
  (empty shot308)
  (empty shot247)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail294 ingredient418)
  (cocktail-part2 cocktail294 ingredient237)
)
 (:goal
  (and
      (contains shot308 cocktail294)
      (contains shot247 cocktail294)
)))
