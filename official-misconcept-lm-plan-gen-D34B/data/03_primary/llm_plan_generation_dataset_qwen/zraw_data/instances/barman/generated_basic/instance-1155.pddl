(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot445 shot420 - shot
      ingredient397 ingredient430 ingredient216 ingredient146 - ingredient
      cocktail1 - cocktail
      dispenser67 dispenser486 dispenser373 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot445)
  (ontable shot420)
  (dispenses dispenser67 ingredient397)
  (dispenses dispenser486 ingredient430)
  (dispenses dispenser373 ingredient216)
  (dispenses dispenser393 ingredient146)
  (clean shaker251)
  (clean shot445)
  (clean shot420)
  (empty shaker251)
  (empty shot445)
  (empty shot420)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient146)
  (cocktail-part2 cocktail1 ingredient430)
)
 (:goal
  (and
      (contains shot445 cocktail1)
)))
