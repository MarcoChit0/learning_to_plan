(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot145 - shot
      ingredient159 ingredient494 ingredient15 ingredient452 - ingredient
      cocktail46 - cocktail
      dispenser14 dispenser189 dispenser156 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot145)
  (dispenses dispenser14 ingredient159)
  (dispenses dispenser189 ingredient494)
  (dispenses dispenser156 ingredient15)
  (dispenses dispenser290 ingredient452)
  (clean shaker143)
  (clean shot145)
  (empty shaker143)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient15)
  (cocktail-part2 cocktail46 ingredient452)
)
 (:goal
  (and
      (contains shot145 cocktail46)
)))
