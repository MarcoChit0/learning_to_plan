(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot145 - shot
      ingredient437 ingredient147 ingredient173 ingredient67 - ingredient
      cocktail380 - cocktail
      dispenser471 dispenser9 dispenser291 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot145)
  (dispenses dispenser471 ingredient437)
  (dispenses dispenser9 ingredient147)
  (dispenses dispenser291 ingredient173)
  (dispenses dispenser269 ingredient67)
  (clean shaker447)
  (clean shot145)
  (empty shaker447)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient67)
  (cocktail-part2 cocktail380 ingredient437)
)
 (:goal
  (and
      (contains shot145 cocktail380)
)))
