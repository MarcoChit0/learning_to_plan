(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot450 shot97 shot205 - shot
      ingredient108 ingredient309 ingredient221 ingredient339 - ingredient
      cocktail466 - cocktail
      dispenser303 dispenser260 dispenser186 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot450)
  (ontable shot97)
  (ontable shot205)
  (dispenses dispenser303 ingredient108)
  (dispenses dispenser260 ingredient309)
  (dispenses dispenser186 ingredient221)
  (dispenses dispenser333 ingredient339)
  (clean shaker249)
  (clean shot450)
  (clean shot97)
  (clean shot205)
  (empty shaker249)
  (empty shot450)
  (empty shot97)
  (empty shot205)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail466 ingredient309)
  (cocktail-part2 cocktail466 ingredient221)
)
 (:goal
  (and
      (contains shot450 cocktail466)
      (contains shot97 cocktail466)
)))
