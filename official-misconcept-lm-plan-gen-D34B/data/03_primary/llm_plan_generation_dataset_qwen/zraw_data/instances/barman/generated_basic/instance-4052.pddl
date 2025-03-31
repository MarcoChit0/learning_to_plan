(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot177 shot229 - shot
      ingredient462 ingredient240 ingredient68 ingredient286 - ingredient
      cocktail336 - cocktail
      dispenser146 dispenser425 dispenser95 dispenser386 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot177)
  (ontable shot229)
  (dispenses dispenser146 ingredient462)
  (dispenses dispenser425 ingredient240)
  (dispenses dispenser95 ingredient68)
  (dispenses dispenser386 ingredient286)
  (clean shaker220)
  (clean shot177)
  (clean shot229)
  (empty shaker220)
  (empty shot177)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail336 ingredient68)
  (cocktail-part2 cocktail336 ingredient462)
)
 (:goal
  (and
      (contains shot177 cocktail336)
)))
