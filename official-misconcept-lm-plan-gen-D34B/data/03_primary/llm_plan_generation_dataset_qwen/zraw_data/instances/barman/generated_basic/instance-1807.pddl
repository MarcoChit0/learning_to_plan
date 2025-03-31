(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot163 shot35 - shot
      ingredient109 ingredient244 ingredient424 ingredient476 - ingredient
      cocktail308 - cocktail
      dispenser488 dispenser80 dispenser356 dispenser38 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot163)
  (ontable shot35)
  (dispenses dispenser488 ingredient109)
  (dispenses dispenser80 ingredient244)
  (dispenses dispenser356 ingredient424)
  (dispenses dispenser38 ingredient476)
  (clean shaker336)
  (clean shot163)
  (clean shot35)
  (empty shaker336)
  (empty shot163)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient476)
  (cocktail-part2 cocktail308 ingredient424)
)
 (:goal
  (and
      (contains shot163 cocktail308)
)))
