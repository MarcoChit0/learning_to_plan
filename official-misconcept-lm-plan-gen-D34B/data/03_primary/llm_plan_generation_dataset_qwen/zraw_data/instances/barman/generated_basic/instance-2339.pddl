(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot109 - shot
      ingredient0 ingredient466 - ingredient
      cocktail266 - cocktail
      dispenser467 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot109)
  (dispenses dispenser467 ingredient0)
  (dispenses dispenser446 ingredient466)
  (clean shaker347)
  (clean shot109)
  (empty shaker347)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient0)
  (cocktail-part2 cocktail266 ingredient466)
)
 (:goal
  (and
      (contains shot109 cocktail266)
)))
