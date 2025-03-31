(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot365 shot379 - shot
      ingredient424 ingredient310 ingredient322 - ingredient
      cocktail411 - cocktail
      dispenser441 dispenser98 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot365)
  (ontable shot379)
  (dispenses dispenser441 ingredient424)
  (dispenses dispenser98 ingredient310)
  (dispenses dispenser282 ingredient322)
  (clean shaker107)
  (clean shot365)
  (clean shot379)
  (empty shaker107)
  (empty shot365)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail411 ingredient310)
  (cocktail-part2 cocktail411 ingredient322)
)
 (:goal
  (and
      (contains shot365 cocktail411)
)))
