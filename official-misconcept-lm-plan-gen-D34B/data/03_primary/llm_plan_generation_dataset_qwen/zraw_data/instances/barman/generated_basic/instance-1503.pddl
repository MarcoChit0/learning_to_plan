(define (problem prob)
 (:domain barman)
 (:objects 
      shaker470 - shaker
      left right - hand
      shot120 shot168 - shot
      ingredient445 ingredient444 ingredient52 ingredient124 - ingredient
      cocktail422 - cocktail
      dispenser218 dispenser131 dispenser64 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker470)
  (ontable shot120)
  (ontable shot168)
  (dispenses dispenser218 ingredient445)
  (dispenses dispenser131 ingredient444)
  (dispenses dispenser64 ingredient52)
  (dispenses dispenser446 ingredient124)
  (clean shaker470)
  (clean shot120)
  (clean shot168)
  (empty shaker470)
  (empty shot120)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker470 l0)
  (shaker-level shaker470 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient445)
  (cocktail-part2 cocktail422 ingredient124)
)
 (:goal
  (and
      (contains shot120 cocktail422)
)))
