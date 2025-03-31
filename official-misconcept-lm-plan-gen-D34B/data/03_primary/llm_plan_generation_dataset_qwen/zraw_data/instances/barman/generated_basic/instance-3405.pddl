(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot244 shot255 - shot
      ingredient235 ingredient139 ingredient439 ingredient454 - ingredient
      cocktail181 - cocktail
      dispenser313 dispenser418 dispenser19 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot244)
  (ontable shot255)
  (dispenses dispenser313 ingredient235)
  (dispenses dispenser418 ingredient139)
  (dispenses dispenser19 ingredient439)
  (dispenses dispenser320 ingredient454)
  (clean shaker322)
  (clean shot244)
  (clean shot255)
  (empty shaker322)
  (empty shot244)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient235)
  (cocktail-part2 cocktail181 ingredient139)
)
 (:goal
  (and
      (contains shot244 cocktail181)
)))
