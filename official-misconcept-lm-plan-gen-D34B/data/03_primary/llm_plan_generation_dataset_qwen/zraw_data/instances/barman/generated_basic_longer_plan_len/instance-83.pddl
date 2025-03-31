(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot120 shot85 shot379 - shot
      ingredient429 ingredient70 ingredient343 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser350 dispenser189 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot120)
  (ontable shot85)
  (ontable shot379)
  (dispenses dispenser350 ingredient429)
  (dispenses dispenser189 ingredient70)
  (dispenses dispenser449 ingredient343)
  (clean shaker250)
  (clean shot120)
  (clean shot85)
  (clean shot379)
  (empty shaker250)
  (empty shot120)
  (empty shot85)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient429)
  (cocktail-part2 cocktail1 ingredient70)
  (cocktail-part1 cocktail2 ingredient343)
  (cocktail-part2 cocktail2 ingredient70)
)
 (:goal
  (and
      (contains shot120 cocktail1)
      (contains shot85 cocktail2)
)))
