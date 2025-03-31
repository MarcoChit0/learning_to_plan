(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot176 shot255 shot120 - shot
      ingredient43 ingredient268 ingredient81 ingredient227 - ingredient
      cocktail75 - cocktail
      dispenser53 dispenser339 dispenser55 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot176)
  (ontable shot255)
  (ontable shot120)
  (dispenses dispenser53 ingredient43)
  (dispenses dispenser339 ingredient268)
  (dispenses dispenser55 ingredient81)
  (dispenses dispenser402 ingredient227)
  (clean shaker216)
  (clean shot176)
  (clean shot255)
  (clean shot120)
  (empty shaker216)
  (empty shot176)
  (empty shot255)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient43)
  (cocktail-part2 cocktail75 ingredient268)
)
 (:goal
  (and
      (contains shot176 cocktail75)
      (contains shot255 cocktail75)
)))
