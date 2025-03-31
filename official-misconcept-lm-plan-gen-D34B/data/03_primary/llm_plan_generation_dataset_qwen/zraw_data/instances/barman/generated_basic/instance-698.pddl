(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot456 shot109 - shot
      ingredient25 ingredient282 ingredient161 - ingredient
      cocktail1 - cocktail
      dispenser456 dispenser417 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot456)
  (ontable shot109)
  (dispenses dispenser456 ingredient25)
  (dispenses dispenser417 ingredient282)
  (dispenses dispenser214 ingredient161)
  (clean shaker438)
  (clean shot456)
  (clean shot109)
  (empty shaker438)
  (empty shot456)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient25)
  (cocktail-part2 cocktail1 ingredient282)
)
 (:goal
  (and
      (contains shot456 cocktail1)
)))
