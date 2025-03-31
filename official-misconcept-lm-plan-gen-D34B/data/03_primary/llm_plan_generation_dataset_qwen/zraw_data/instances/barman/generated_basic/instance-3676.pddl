(define (problem prob)
 (:domain barman)
 (:objects 
      shaker387 - shaker
      left right - hand
      shot357 shot296 shot268 - shot
      ingredient91 ingredient171 ingredient289 ingredient434 - ingredient
      cocktail396 - cocktail
      dispenser313 dispenser388 dispenser454 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker387)
  (ontable shot357)
  (ontable shot296)
  (ontable shot268)
  (dispenses dispenser313 ingredient91)
  (dispenses dispenser388 ingredient171)
  (dispenses dispenser454 ingredient289)
  (dispenses dispenser273 ingredient434)
  (clean shaker387)
  (clean shot357)
  (clean shot296)
  (clean shot268)
  (empty shaker387)
  (empty shot357)
  (empty shot296)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker387 l0)
  (shaker-level shaker387 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail396 ingredient171)
  (cocktail-part2 cocktail396 ingredient289)
)
 (:goal
  (and
      (contains shot357 cocktail396)
      (contains shot296 ingredient289)
)))
