(define (problem prob)
 (:domain barman)
 (:objects 
      shaker130 - shaker
      left right - hand
      shot210 shot307 - shot
      ingredient289 ingredient98 - ingredient
      cocktail1 - cocktail
      dispenser296 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker130)
  (ontable shot210)
  (ontable shot307)
  (dispenses dispenser296 ingredient289)
  (dispenses dispenser362 ingredient98)
  (clean shaker130)
  (clean shot210)
  (clean shot307)
  (empty shaker130)
  (empty shot210)
  (empty shot307)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker130 l0)
  (shaker-level shaker130 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient98)
  (cocktail-part2 cocktail1 ingredient289)
)
 (:goal
  (and
      (contains shot210 cocktail1)
)))
