(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot103 shot166 - shot
      ingredient222 ingredient273 ingredient221 - ingredient
      cocktail1 - cocktail
      dispenser169 dispenser201 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot103)
  (ontable shot166)
  (dispenses dispenser169 ingredient222)
  (dispenses dispenser201 ingredient273)
  (dispenses dispenser421 ingredient221)
  (clean shaker488)
  (clean shot103)
  (clean shot166)
  (empty shaker488)
  (empty shot103)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient222)
  (cocktail-part2 cocktail1 ingredient221)
)
 (:goal
  (and
      (contains shot103 cocktail1)
)))
