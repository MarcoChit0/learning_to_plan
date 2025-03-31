(define (problem prob)
 (:domain barman)
 (:objects 
      shaker335 - shaker
      left right - hand
      shot103 shot253 shot150 - shot
      ingredient64 ingredient184 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser315 dispenser75 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker335)
  (ontable shot103)
  (ontable shot253)
  (ontable shot150)
  (dispenses dispenser315 ingredient64)
  (dispenses dispenser75 ingredient184)
  (clean shaker335)
  (clean shot103)
  (clean shot253)
  (clean shot150)
  (empty shaker335)
  (empty shot103)
  (empty shot253)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker335 l0)
  (shaker-level shaker335 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient64)
  (cocktail-part1 cocktail2 ingredient184)
  (cocktail-part2 cocktail2 ingredient64)
)
 (:goal
  (and
      (contains shot103 cocktail1)
      (contains shot253 cocktail2)
)))
