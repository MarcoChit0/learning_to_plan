(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot221 - shot
      ingredient244 ingredient365 - ingredient
      cocktail307 - cocktail
      dispenser231 dispenser401 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot221)
  (dispenses dispenser231 ingredient244)
  (dispenses dispenser401 ingredient365)
  (clean shaker168)
  (clean shot221)
  (empty shaker168)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient365)
  (cocktail-part2 cocktail307 ingredient244)
)
 (:goal
  (and
      (contains shot221 cocktail307)
)))
