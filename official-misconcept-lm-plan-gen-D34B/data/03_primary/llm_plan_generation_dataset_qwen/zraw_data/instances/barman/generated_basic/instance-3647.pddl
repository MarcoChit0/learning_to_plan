(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot278 shot403 - shot
      ingredient481 ingredient229 ingredient151 ingredient365 - ingredient
      cocktail439 - cocktail
      dispenser409 dispenser174 dispenser42 dispenser159 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot278)
  (ontable shot403)
  (dispenses dispenser409 ingredient481)
  (dispenses dispenser174 ingredient229)
  (dispenses dispenser42 ingredient151)
  (dispenses dispenser159 ingredient365)
  (clean shaker16)
  (clean shot278)
  (clean shot403)
  (empty shaker16)
  (empty shot278)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient229)
  (cocktail-part2 cocktail439 ingredient365)
)
 (:goal
  (and
      (contains shot278 cocktail439)
)))
