(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot89 shot347 - shot
      ingredient292 ingredient86 ingredient79 - ingredient
      cocktail144 - cocktail
      dispenser421 dispenser138 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot89)
  (ontable shot347)
  (dispenses dispenser421 ingredient292)
  (dispenses dispenser138 ingredient86)
  (dispenses dispenser259 ingredient79)
  (clean shaker365)
  (clean shot89)
  (clean shot347)
  (empty shaker365)
  (empty shot89)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail144 ingredient86)
  (cocktail-part2 cocktail144 ingredient79)
)
 (:goal
  (and
      (contains shot89 cocktail144)
)))
