(define (problem prob)
 (:domain barman)
 (:objects 
      shaker444 - shaker
      left right - hand
      shot274 shot398 shot259 - shot
      ingredient244 ingredient12 ingredient96 ingredient305 - ingredient
      cocktail430 - cocktail
      dispenser399 dispenser228 dispenser347 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker444)
  (ontable shot274)
  (ontable shot398)
  (ontable shot259)
  (dispenses dispenser399 ingredient244)
  (dispenses dispenser228 ingredient12)
  (dispenses dispenser347 ingredient96)
  (dispenses dispenser425 ingredient305)
  (clean shaker444)
  (clean shot274)
  (clean shot398)
  (clean shot259)
  (empty shaker444)
  (empty shot274)
  (empty shot398)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker444 l0)
  (shaker-level shaker444 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient305)
  (cocktail-part2 cocktail430 ingredient96)
)
 (:goal
  (and
      (contains shot274 cocktail430)
      (contains shot398 cocktail430)
)))
