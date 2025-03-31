(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot67 - shot
      ingredient76 ingredient212 ingredient187 - ingredient
      cocktail102 - cocktail
      dispenser433 dispenser256 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot67)
  (dispenses dispenser433 ingredient76)
  (dispenses dispenser256 ingredient212)
  (dispenses dispenser272 ingredient187)
  (clean shaker281)
  (clean shot67)
  (empty shaker281)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail102 ingredient76)
  (cocktail-part2 cocktail102 ingredient187)
)
 (:goal
  (and
      (contains shot67 cocktail102)
)))
