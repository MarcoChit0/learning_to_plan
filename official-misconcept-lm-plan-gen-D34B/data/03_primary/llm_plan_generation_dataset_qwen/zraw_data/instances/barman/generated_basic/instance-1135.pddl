(define (problem prob)
 (:domain barman)
 (:objects 
      shaker389 - shaker
      left right - hand
      shot45 shot34 - shot
      ingredient407 ingredient73 ingredient149 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser149 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker389)
  (ontable shot45)
  (ontable shot34)
  (dispenses dispenser42 ingredient407)
  (dispenses dispenser149 ingredient73)
  (dispenses dispenser173 ingredient149)
  (clean shaker389)
  (clean shot45)
  (clean shot34)
  (empty shaker389)
  (empty shot45)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker389 l0)
  (shaker-level shaker389 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient149)
  (cocktail-part2 cocktail1 ingredient73)
)
 (:goal
  (and
      (contains shot45 cocktail1)
)))
