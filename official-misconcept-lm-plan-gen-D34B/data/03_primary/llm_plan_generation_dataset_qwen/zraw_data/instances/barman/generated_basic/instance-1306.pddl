(define (problem prob)
 (:domain barman)
 (:objects 
      shaker258 - shaker
      left right - hand
      shot79 shot26 shot277 - shot
      ingredient399 ingredient170 - ingredient
      cocktail435 - cocktail
      dispenser8 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker258)
  (ontable shot79)
  (ontable shot26)
  (ontable shot277)
  (dispenses dispenser8 ingredient399)
  (dispenses dispenser414 ingredient170)
  (clean shaker258)
  (clean shot79)
  (clean shot26)
  (clean shot277)
  (empty shaker258)
  (empty shot79)
  (empty shot26)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker258 l0)
  (shaker-level shaker258 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail435 ingredient170)
  (cocktail-part2 cocktail435 ingredient399)
)
 (:goal
  (and
      (contains shot79 cocktail435)
      (contains shot26 ingredient170)
)))
