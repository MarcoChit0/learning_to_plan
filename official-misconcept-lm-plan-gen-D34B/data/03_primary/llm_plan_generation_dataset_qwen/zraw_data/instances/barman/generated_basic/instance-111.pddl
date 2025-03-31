(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot423 shot223 shot444 - shot
      ingredient399 ingredient402 - ingredient
      cocktail1 - cocktail
      dispenser252 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot423)
  (ontable shot223)
  (ontable shot444)
  (dispenses dispenser252 ingredient399)
  (dispenses dispenser319 ingredient402)
  (clean shaker77)
  (clean shot423)
  (clean shot223)
  (clean shot444)
  (empty shaker77)
  (empty shot423)
  (empty shot223)
  (empty shot444)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient402)
  (cocktail-part2 cocktail1 ingredient399)
)
 (:goal
  (and
      (contains shot423 cocktail1)
      (contains shot223 ingredient399)
)))
