(define (problem prob)
 (:domain barman)
 (:objects 
      shaker111 - shaker
      left right - hand
      shot259 shot365 - shot
      ingredient172 ingredient400 ingredient458 ingredient177 - ingredient
      cocktail348 - cocktail
      dispenser362 dispenser224 dispenser239 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker111)
  (ontable shot259)
  (ontable shot365)
  (dispenses dispenser362 ingredient172)
  (dispenses dispenser224 ingredient400)
  (dispenses dispenser239 ingredient458)
  (dispenses dispenser4 ingredient177)
  (clean shaker111)
  (clean shot259)
  (clean shot365)
  (empty shaker111)
  (empty shot259)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker111 l0)
  (shaker-level shaker111 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail348 ingredient172)
  (cocktail-part2 cocktail348 ingredient400)
)
 (:goal
  (and
      (contains shot259 cocktail348)
)))
