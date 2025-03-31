(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot283 shot137 shot221 - shot
      ingredient216 ingredient455 ingredient487 ingredient439 - ingredient
      cocktail388 - cocktail
      dispenser49 dispenser159 dispenser117 dispenser185 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot283)
  (ontable shot137)
  (ontable shot221)
  (dispenses dispenser49 ingredient216)
  (dispenses dispenser159 ingredient455)
  (dispenses dispenser117 ingredient487)
  (dispenses dispenser185 ingredient439)
  (clean shaker241)
  (clean shot283)
  (clean shot137)
  (clean shot221)
  (empty shaker241)
  (empty shot283)
  (empty shot137)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient487)
  (cocktail-part2 cocktail388 ingredient455)
)
 (:goal
  (and
      (contains shot283 cocktail388)
      (contains shot137 cocktail388)
)))
