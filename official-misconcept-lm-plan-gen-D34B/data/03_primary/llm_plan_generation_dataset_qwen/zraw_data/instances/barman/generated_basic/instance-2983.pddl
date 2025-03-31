(define (problem prob)
 (:domain barman)
 (:objects 
      shaker217 - shaker
      left right - hand
      shot31 shot59 - shot
      ingredient174 ingredient469 ingredient186 ingredient217 - ingredient
      cocktail408 - cocktail
      dispenser459 dispenser170 dispenser199 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker217)
  (ontable shot31)
  (ontable shot59)
  (dispenses dispenser459 ingredient174)
  (dispenses dispenser170 ingredient469)
  (dispenses dispenser199 ingredient186)
  (dispenses dispenser313 ingredient217)
  (clean shaker217)
  (clean shot31)
  (clean shot59)
  (empty shaker217)
  (empty shot31)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker217 l0)
  (shaker-level shaker217 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail408 ingredient186)
  (cocktail-part2 cocktail408 ingredient469)
)
 (:goal
  (and
      (contains shot31 cocktail408)
)))
