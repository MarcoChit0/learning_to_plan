(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot374 shot78 - shot
      ingredient299 ingredient466 - ingredient
      cocktail438 - cocktail
      dispenser385 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot374)
  (ontable shot78)
  (dispenses dispenser385 ingredient299)
  (dispenses dispenser97 ingredient466)
  (clean shaker450)
  (clean shot374)
  (clean shot78)
  (empty shaker450)
  (empty shot374)
  (empty shot78)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail438 ingredient299)
  (cocktail-part2 cocktail438 ingredient466)
)
 (:goal
  (and
      (contains shot374 cocktail438)
)))
