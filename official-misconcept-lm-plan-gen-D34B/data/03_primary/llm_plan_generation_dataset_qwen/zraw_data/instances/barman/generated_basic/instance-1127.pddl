(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot225 shot16 - shot
      ingredient301 ingredient350 ingredient5 - ingredient
      cocktail1 - cocktail
      dispenser179 dispenser6 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot225)
  (ontable shot16)
  (dispenses dispenser179 ingredient301)
  (dispenses dispenser6 ingredient350)
  (dispenses dispenser52 ingredient5)
  (clean shaker136)
  (clean shot225)
  (clean shot16)
  (empty shaker136)
  (empty shot225)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient5)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot225 cocktail1)
)))
