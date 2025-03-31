(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot195 shot161 - shot
      ingredient388 ingredient186 ingredient363 ingredient473 - ingredient
      cocktail1 - cocktail
      dispenser268 dispenser188 dispenser295 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot195)
  (ontable shot161)
  (dispenses dispenser268 ingredient388)
  (dispenses dispenser188 ingredient186)
  (dispenses dispenser295 ingredient363)
  (dispenses dispenser175 ingredient473)
  (clean shaker9)
  (clean shot195)
  (clean shot161)
  (empty shaker9)
  (empty shot195)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient388)
  (cocktail-part2 cocktail1 ingredient473)
)
 (:goal
  (and
      (contains shot195 cocktail1)
)))
