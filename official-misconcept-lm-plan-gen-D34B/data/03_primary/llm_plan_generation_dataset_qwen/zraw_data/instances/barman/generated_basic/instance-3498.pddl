(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot330 shot405 - shot
      ingredient211 ingredient363 ingredient275 ingredient372 - ingredient
      cocktail266 - cocktail
      dispenser63 dispenser138 dispenser243 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot330)
  (ontable shot405)
  (dispenses dispenser63 ingredient211)
  (dispenses dispenser138 ingredient363)
  (dispenses dispenser243 ingredient275)
  (dispenses dispenser391 ingredient372)
  (clean shaker317)
  (clean shot330)
  (clean shot405)
  (empty shaker317)
  (empty shot330)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient211)
  (cocktail-part2 cocktail266 ingredient275)
)
 (:goal
  (and
      (contains shot330 cocktail266)
)))
