(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot349 shot396 - shot
      ingredient356 ingredient90 ingredient366 - ingredient
      cocktail206 - cocktail
      dispenser279 dispenser423 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot349)
  (ontable shot396)
  (dispenses dispenser279 ingredient356)
  (dispenses dispenser423 ingredient90)
  (dispenses dispenser116 ingredient366)
  (clean shaker352)
  (clean shot349)
  (clean shot396)
  (empty shaker352)
  (empty shot349)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient366)
  (cocktail-part2 cocktail206 ingredient356)
)
 (:goal
  (and
      (contains shot349 cocktail206)
)))
