(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot218 shot175 - shot
      ingredient463 ingredient161 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser9 dispenser362 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot218)
  (ontable shot175)
  (dispenses dispenser9 ingredient463)
  (dispenses dispenser362 ingredient161)
  (dispenses dispenser162 ingredient390)
  (clean shaker178)
  (clean shot218)
  (clean shot175)
  (empty shaker178)
  (empty shot218)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient161)
  (cocktail-part2 cocktail1 ingredient463)
)
 (:goal
  (and
      (contains shot218 cocktail1)
)))
