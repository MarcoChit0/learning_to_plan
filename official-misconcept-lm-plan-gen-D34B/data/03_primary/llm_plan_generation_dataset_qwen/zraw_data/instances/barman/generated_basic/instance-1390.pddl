(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot304 shot41 shot297 - shot
      ingredient339 ingredient221 ingredient318 ingredient187 - ingredient
      cocktail273 - cocktail
      dispenser265 dispenser284 dispenser46 dispenser454 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot304)
  (ontable shot41)
  (ontable shot297)
  (dispenses dispenser265 ingredient339)
  (dispenses dispenser284 ingredient221)
  (dispenses dispenser46 ingredient318)
  (dispenses dispenser454 ingredient187)
  (clean shaker321)
  (clean shot304)
  (clean shot41)
  (clean shot297)
  (empty shaker321)
  (empty shot304)
  (empty shot41)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient187)
  (cocktail-part2 cocktail273 ingredient339)
)
 (:goal
  (and
      (contains shot304 cocktail273)
      (contains shot41 ingredient221)
)))
