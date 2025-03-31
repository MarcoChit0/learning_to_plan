(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot105 shot250 - shot
      ingredient135 ingredient317 ingredient81 ingredient132 - ingredient
      cocktail303 - cocktail
      dispenser488 dispenser429 dispenser84 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot105)
  (ontable shot250)
  (dispenses dispenser488 ingredient135)
  (dispenses dispenser429 ingredient317)
  (dispenses dispenser84 ingredient81)
  (dispenses dispenser348 ingredient132)
  (clean shaker4)
  (clean shot105)
  (clean shot250)
  (empty shaker4)
  (empty shot105)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail303 ingredient135)
  (cocktail-part2 cocktail303 ingredient81)
)
 (:goal
  (and
      (contains shot105 cocktail303)
)))
