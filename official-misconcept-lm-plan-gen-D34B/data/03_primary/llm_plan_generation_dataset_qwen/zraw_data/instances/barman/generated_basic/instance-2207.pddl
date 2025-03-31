(define (problem prob)
 (:domain barman)
 (:objects 
      shaker100 - shaker
      left right - hand
      shot364 shot348 shot376 - shot
      ingredient469 ingredient193 ingredient294 ingredient23 - ingredient
      cocktail161 - cocktail
      dispenser251 dispenser296 dispenser275 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker100)
  (ontable shot364)
  (ontable shot348)
  (ontable shot376)
  (dispenses dispenser251 ingredient469)
  (dispenses dispenser296 ingredient193)
  (dispenses dispenser275 ingredient294)
  (dispenses dispenser193 ingredient23)
  (clean shaker100)
  (clean shot364)
  (clean shot348)
  (clean shot376)
  (empty shaker100)
  (empty shot364)
  (empty shot348)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker100 l0)
  (shaker-level shaker100 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail161 ingredient294)
  (cocktail-part2 cocktail161 ingredient23)
)
 (:goal
  (and
      (contains shot364 cocktail161)
      (contains shot348 ingredient469)
)))
