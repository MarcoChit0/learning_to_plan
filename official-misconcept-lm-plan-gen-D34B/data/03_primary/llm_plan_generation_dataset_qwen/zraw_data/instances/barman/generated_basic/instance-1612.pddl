(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot389 shot340 - shot
      ingredient56 ingredient489 ingredient102 - ingredient
      cocktail126 - cocktail
      dispenser426 dispenser459 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot389)
  (ontable shot340)
  (dispenses dispenser426 ingredient56)
  (dispenses dispenser459 ingredient489)
  (dispenses dispenser149 ingredient102)
  (clean shaker230)
  (clean shot389)
  (clean shot340)
  (empty shaker230)
  (empty shot389)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient102)
  (cocktail-part2 cocktail126 ingredient489)
)
 (:goal
  (and
      (contains shot389 cocktail126)
)))
