(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot434 shot377 shot81 - shot
      ingredient428 ingredient262 ingredient300 ingredient132 - ingredient
      cocktail430 - cocktail
      dispenser270 dispenser317 dispenser314 dispenser371 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot434)
  (ontable shot377)
  (ontable shot81)
  (dispenses dispenser270 ingredient428)
  (dispenses dispenser317 ingredient262)
  (dispenses dispenser314 ingredient300)
  (dispenses dispenser371 ingredient132)
  (clean shaker294)
  (clean shot434)
  (clean shot377)
  (clean shot81)
  (empty shaker294)
  (empty shot434)
  (empty shot377)
  (empty shot81)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient262)
  (cocktail-part2 cocktail430 ingredient132)
)
 (:goal
  (and
      (contains shot434 cocktail430)
      (contains shot377 cocktail430)
)))
