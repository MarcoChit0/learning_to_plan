(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot16 shot417 shot257 shot342 - shot
      ingredient145 ingredient380 ingredient179 - ingredient
      cocktail369 - cocktail
      dispenser181 dispenser81 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot16)
  (ontable shot417)
  (ontable shot257)
  (ontable shot342)
  (dispenses dispenser181 ingredient145)
  (dispenses dispenser81 ingredient380)
  (dispenses dispenser273 ingredient179)
  (clean shaker383)
  (clean shot16)
  (clean shot417)
  (clean shot257)
  (clean shot342)
  (empty shaker383)
  (empty shot16)
  (empty shot417)
  (empty shot257)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail369 ingredient380)
  (cocktail-part2 cocktail369 ingredient179)
)
 (:goal
  (and
      (contains shot16 cocktail369)
      (contains shot417 cocktail369)
      (contains shot257 ingredient145)
)))
