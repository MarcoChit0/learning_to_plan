(define (problem prob)
 (:domain barman)
 (:objects 
      shaker285 - shaker
      left right - hand
      shot363 shot438 shot285 - shot
      ingredient493 ingredient346 ingredient277 ingredient427 - ingredient
      cocktail461 - cocktail
      dispenser117 dispenser48 dispenser275 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker285)
  (ontable shot363)
  (ontable shot438)
  (ontable shot285)
  (dispenses dispenser117 ingredient493)
  (dispenses dispenser48 ingredient346)
  (dispenses dispenser275 ingredient277)
  (dispenses dispenser151 ingredient427)
  (clean shaker285)
  (clean shot363)
  (clean shot438)
  (clean shot285)
  (empty shaker285)
  (empty shot363)
  (empty shot438)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker285 l0)
  (shaker-level shaker285 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient346)
  (cocktail-part2 cocktail461 ingredient493)
)
 (:goal
  (and
      (contains shot363 cocktail461)
      (contains shot438 ingredient427)
)))
