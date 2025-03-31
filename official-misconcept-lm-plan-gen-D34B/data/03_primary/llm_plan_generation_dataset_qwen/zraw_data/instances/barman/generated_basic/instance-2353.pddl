(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot137 shot348 - shot
      ingredient144 ingredient489 ingredient174 ingredient177 - ingredient
      cocktail160 - cocktail
      dispenser318 dispenser172 dispenser446 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot137)
  (ontable shot348)
  (dispenses dispenser318 ingredient144)
  (dispenses dispenser172 ingredient489)
  (dispenses dispenser446 ingredient174)
  (dispenses dispenser272 ingredient177)
  (clean shaker184)
  (clean shot137)
  (clean shot348)
  (empty shaker184)
  (empty shot137)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail160 ingredient177)
  (cocktail-part2 cocktail160 ingredient489)
)
 (:goal
  (and
      (contains shot137 cocktail160)
)))
