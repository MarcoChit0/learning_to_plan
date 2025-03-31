(define (problem prob)
 (:domain barman)
 (:objects 
      shaker147 - shaker
      left right - hand
      shot67 shot88 - shot
      ingredient346 ingredient462 ingredient490 ingredient339 - ingredient
      cocktail343 - cocktail
      dispenser395 dispenser32 dispenser71 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker147)
  (ontable shot67)
  (ontable shot88)
  (dispenses dispenser395 ingredient346)
  (dispenses dispenser32 ingredient462)
  (dispenses dispenser71 ingredient490)
  (dispenses dispenser466 ingredient339)
  (clean shaker147)
  (clean shot67)
  (clean shot88)
  (empty shaker147)
  (empty shot67)
  (empty shot88)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker147 l0)
  (shaker-level shaker147 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail343 ingredient346)
  (cocktail-part2 cocktail343 ingredient339)
)
 (:goal
  (and
      (contains shot67 cocktail343)
)))
