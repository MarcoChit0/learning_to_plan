(define (problem prob)
 (:domain barman)
 (:objects 
      shaker97 - shaker
      left right - hand
      shot229 shot365 - shot
      ingredient96 ingredient138 ingredient207 - ingredient
      cocktail430 - cocktail
      dispenser120 dispenser365 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker97)
  (ontable shot229)
  (ontable shot365)
  (dispenses dispenser120 ingredient96)
  (dispenses dispenser365 ingredient138)
  (dispenses dispenser345 ingredient207)
  (clean shaker97)
  (clean shot229)
  (clean shot365)
  (empty shaker97)
  (empty shot229)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker97 l0)
  (shaker-level shaker97 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient207)
  (cocktail-part2 cocktail430 ingredient96)
)
 (:goal
  (and
      (contains shot229 cocktail430)
)))
