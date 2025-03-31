(define (problem prob)
 (:domain barman)
 (:objects 
      shaker223 - shaker
      left right - hand
      shot472 shot145 - shot
      ingredient175 ingredient369 ingredient14 ingredient223 - ingredient
      cocktail74 - cocktail
      dispenser402 dispenser49 dispenser433 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker223)
  (ontable shot472)
  (ontable shot145)
  (dispenses dispenser402 ingredient175)
  (dispenses dispenser49 ingredient369)
  (dispenses dispenser433 ingredient14)
  (dispenses dispenser57 ingredient223)
  (clean shaker223)
  (clean shot472)
  (clean shot145)
  (empty shaker223)
  (empty shot472)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker223 l0)
  (shaker-level shaker223 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient14)
  (cocktail-part2 cocktail74 ingredient369)
)
 (:goal
  (and
      (contains shot472 cocktail74)
)))
