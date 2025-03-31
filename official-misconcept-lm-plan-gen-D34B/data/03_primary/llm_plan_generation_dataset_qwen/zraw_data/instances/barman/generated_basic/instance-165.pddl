(define (problem prob)
 (:domain barman)
 (:objects 
      shaker71 - shaker
      left right - hand
      shot186 shot168 - shot
      ingredient358 ingredient289 ingredient67 - ingredient
      cocktail1 - cocktail
      dispenser404 dispenser343 dispenser265 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker71)
  (ontable shot186)
  (ontable shot168)
  (dispenses dispenser404 ingredient358)
  (dispenses dispenser343 ingredient289)
  (dispenses dispenser265 ingredient67)
  (clean shaker71)
  (clean shot186)
  (clean shot168)
  (empty shaker71)
  (empty shot186)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker71 l0)
  (shaker-level shaker71 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient358)
)
 (:goal
  (and
      (contains shot186 cocktail1)
)))
