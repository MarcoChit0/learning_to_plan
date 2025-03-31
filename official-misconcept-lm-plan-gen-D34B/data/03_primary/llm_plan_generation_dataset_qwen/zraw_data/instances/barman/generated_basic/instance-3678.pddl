(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot212 shot418 shot279 - shot
      ingredient234 ingredient436 ingredient445 ingredient369 - ingredient
      cocktail70 - cocktail
      dispenser460 dispenser147 dispenser280 dispenser304 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot212)
  (ontable shot418)
  (ontable shot279)
  (dispenses dispenser460 ingredient234)
  (dispenses dispenser147 ingredient436)
  (dispenses dispenser280 ingredient445)
  (dispenses dispenser304 ingredient369)
  (clean shaker350)
  (clean shot212)
  (clean shot418)
  (clean shot279)
  (empty shaker350)
  (empty shot212)
  (empty shot418)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient234)
  (cocktail-part2 cocktail70 ingredient369)
)
 (:goal
  (and
      (contains shot212 cocktail70)
      (contains shot418 ingredient445)
)))
