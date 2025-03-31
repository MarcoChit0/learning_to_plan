(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot151 - shot
      ingredient278 ingredient468 - ingredient
      cocktail447 - cocktail
      dispenser346 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot151)
  (dispenses dispenser346 ingredient278)
  (dispenses dispenser203 ingredient468)
  (clean shaker431)
  (clean shot151)
  (empty shaker431)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail447 ingredient278)
  (cocktail-part2 cocktail447 ingredient468)
)
 (:goal
  (and
      (contains shot151 cocktail447)
)))
