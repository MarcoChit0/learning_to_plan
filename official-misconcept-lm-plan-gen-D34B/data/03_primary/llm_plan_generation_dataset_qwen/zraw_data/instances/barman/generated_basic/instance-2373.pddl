(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot361 - shot
      ingredient19 ingredient455 ingredient402 ingredient127 - ingredient
      cocktail344 - cocktail
      dispenser345 dispenser105 dispenser303 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot361)
  (dispenses dispenser345 ingredient19)
  (dispenses dispenser105 ingredient455)
  (dispenses dispenser303 ingredient402)
  (dispenses dispenser331 ingredient127)
  (clean shaker125)
  (clean shot361)
  (empty shaker125)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail344 ingredient127)
  (cocktail-part2 cocktail344 ingredient455)
)
 (:goal
  (and
      (contains shot361 cocktail344)
)))
