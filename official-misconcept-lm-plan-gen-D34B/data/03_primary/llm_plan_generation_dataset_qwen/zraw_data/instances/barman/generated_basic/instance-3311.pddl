(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot357 shot442 shot306 - shot
      ingredient254 ingredient451 ingredient105 - ingredient
      cocktail420 - cocktail
      dispenser117 dispenser335 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot357)
  (ontable shot442)
  (ontable shot306)
  (dispenses dispenser117 ingredient254)
  (dispenses dispenser335 ingredient451)
  (dispenses dispenser277 ingredient105)
  (clean shaker255)
  (clean shot357)
  (clean shot442)
  (clean shot306)
  (empty shaker255)
  (empty shot357)
  (empty shot442)
  (empty shot306)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail420 ingredient105)
  (cocktail-part2 cocktail420 ingredient254)
)
 (:goal
  (and
      (contains shot357 cocktail420)
      (contains shot442 ingredient254)
)))
