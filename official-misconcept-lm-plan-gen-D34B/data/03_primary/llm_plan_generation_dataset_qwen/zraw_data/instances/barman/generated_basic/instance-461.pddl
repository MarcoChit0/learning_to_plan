(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot63 shot420 shot374 - shot
      ingredient489 ingredient45 ingredient309 ingredient274 - ingredient
      cocktail1 - cocktail
      dispenser226 dispenser79 dispenser400 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot63)
  (ontable shot420)
  (ontable shot374)
  (dispenses dispenser226 ingredient489)
  (dispenses dispenser79 ingredient45)
  (dispenses dispenser400 ingredient309)
  (dispenses dispenser406 ingredient274)
  (clean shaker207)
  (clean shot63)
  (clean shot420)
  (clean shot374)
  (empty shaker207)
  (empty shot63)
  (empty shot420)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient489)
  (cocktail-part2 cocktail1 ingredient274)
)
 (:goal
  (and
      (contains shot63 cocktail1)
      (contains shot420 cocktail1)
)))
