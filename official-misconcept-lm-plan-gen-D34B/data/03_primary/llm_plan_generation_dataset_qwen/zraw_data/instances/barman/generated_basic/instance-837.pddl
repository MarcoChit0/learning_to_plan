(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot283 shot402 shot216 - shot
      ingredient356 ingredient485 ingredient291 ingredient108 - ingredient
      cocktail1 - cocktail
      dispenser292 dispenser421 dispenser329 dispenser317 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot283)
  (ontable shot402)
  (ontable shot216)
  (dispenses dispenser292 ingredient356)
  (dispenses dispenser421 ingredient485)
  (dispenses dispenser329 ingredient291)
  (dispenses dispenser317 ingredient108)
  (clean shaker328)
  (clean shot283)
  (clean shot402)
  (clean shot216)
  (empty shaker328)
  (empty shot283)
  (empty shot402)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient108)
  (cocktail-part2 cocktail1 ingredient356)
)
 (:goal
  (and
      (contains shot283 cocktail1)
      (contains shot402 cocktail1)
)))
