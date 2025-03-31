(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot184 shot64 - shot
      ingredient108 ingredient326 - ingredient
      cocktail2 - cocktail
      dispenser149 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot184)
  (ontable shot64)
  (dispenses dispenser149 ingredient108)
  (dispenses dispenser464 ingredient326)
  (clean shaker372)
  (clean shot184)
  (clean shot64)
  (empty shaker372)
  (empty shot184)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient108)
  (cocktail-part2 cocktail2 ingredient326)
)
 (:goal
  (and
      (contains shot184 cocktail2)
)))
