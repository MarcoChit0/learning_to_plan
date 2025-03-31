(define (problem prob)
 (:domain barman)
 (:objects 
      shaker226 - shaker
      left right - hand
      shot204 - shot
      ingredient173 ingredient73 ingredient149 ingredient127 - ingredient
      cocktail1 - cocktail
      dispenser171 dispenser389 dispenser272 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker226)
  (ontable shot204)
  (dispenses dispenser171 ingredient173)
  (dispenses dispenser389 ingredient73)
  (dispenses dispenser272 ingredient149)
  (dispenses dispenser352 ingredient127)
  (clean shaker226)
  (clean shot204)
  (empty shaker226)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker226 l0)
  (shaker-level shaker226 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient127)
  (cocktail-part2 cocktail1 ingredient173)
)
 (:goal
  (and
      (contains shot204 cocktail1)
)))
