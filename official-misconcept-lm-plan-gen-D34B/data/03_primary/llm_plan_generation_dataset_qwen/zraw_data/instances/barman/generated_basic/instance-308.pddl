(define (problem prob)
 (:domain barman)
 (:objects 
      shaker258 - shaker
      left right - hand
      shot496 shot221 - shot
      ingredient402 ingredient377 - ingredient
      cocktail1 - cocktail
      dispenser112 dispenser438 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker258)
  (ontable shot496)
  (ontable shot221)
  (dispenses dispenser112 ingredient402)
  (dispenses dispenser438 ingredient377)
  (clean shaker258)
  (clean shot496)
  (clean shot221)
  (empty shaker258)
  (empty shot496)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker258 l0)
  (shaker-level shaker258 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient377)
  (cocktail-part2 cocktail1 ingredient402)
)
 (:goal
  (and
      (contains shot496 cocktail1)
)))
